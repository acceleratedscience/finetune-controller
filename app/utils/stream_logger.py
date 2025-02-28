import asyncio
from kubernetes.client.rest import ApiException
from fastapi import WebSocket, WebSocketDisconnect
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from app.database.db import db_manager
from app.schemas.db_schemas import DatabaseStatusEnum, JobStatus
from app.schemas.kubeflow_schemas import TrainingJobStatus
from app.utils.kf_config import kubeflow_api
from app.core.config import settings
from app.utils.kube_config import core_v1_api
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class LogStreamManager:
    def __init__(
        self,
        websocket: WebSocket,
        job_id: str,
        full_log: bool,
        follow: bool,
        last_lines: int = 100,
        search_string: str = "Epoch",
    ):
        self.websocket = websocket
        self.job_id = job_id
        self.full_log = full_log
        self.follow_log = follow
        self.last_lines = last_lines
        self.pod_name: str | None = None
        self.container_name = "pytorch"
        self.chunk_size = 4096  # Optimal chunk size for streaming
        self.search_string = search_string
        self.epoch_found = False
        self.is_connected = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def get_job_status(self) -> JobStatus | None:
        """Fetch job status with retry mechanism"""
        return await db_manager.get_job(self.job_id)

    async def wait_for_job_start(self) -> bool:
        """Wait for job to start with timeout"""
        MAX_WAIT_TIME = 300  # 5 minutes timeout
        start_time = asyncio.get_event_loop().time()

        while self.is_connected:
            try:
                job_info = await self.get_job_status()

                if not job_info:
                    await self.send_message("Error: Job not found")
                    return False

                if job_info.status not in TrainingJobStatus.running_states:
                    await self.send_message("Error: Job not in running state")
                    return False

                if job_info.status == DatabaseStatusEnum.running:
                    return True

                if asyncio.get_event_loop().time() - start_time > MAX_WAIT_TIME:
                    await self.send_message("Error: Timeout waiting for job to start")
                    return False

                await self.send_message("Info: Waiting for job to start training...")
                await asyncio.sleep(10)

            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket disconnected while waiting for job {self.job_id} to start"
                )
                self.is_connected = False
                return False
            except Exception as e:
                logger.error(
                    f"Error while waiting for job {self.job_id} to start: {str(e)}"
                )
                await self.send_message(
                    f"Error while waiting for job to start: {str(e)}"
                )
                return False

    async def send_message(self, message: str):
        """Safe message sending with error handling"""
        if not self.is_connected:
            raise WebSocketDisconnect()

        try:
            await self.websocket.send_text(message)
        except Exception as e:
            self.is_connected = False
            raise WebSocketDisconnect()

    def get_pod_name(self) -> str:
        """Get pod name with validation"""
        try:
            pod_names = kubeflow_api.get_job_pod_names(
                self.job_id, namespace=settings.NAMESPACE, is_master=True
            )

            if isinstance(pod_names, list) and pod_names:
                return pod_names[0]
            elif isinstance(pod_names, str) and pod_names:
                return pod_names

            raise ValueError(f"No pod names found for job {self.job_id}")
        except Exception as e:
            logger.error(f"Error getting pod name for job {self.job_id}: {str(e)}")
            raise ValueError(f"Failed to get pod name: {str(e)}")

    async def is_pod_running(self, pod_name: str, namespace: str) -> bool:
        """Check if the pod is still running."""
        try:
            pod = core_v1_api.read_namespaced_pod_status(pod_name, namespace)
            return pod.status.phase in ["Running", "Pending"]
        except ApiException as e:
            logger.warning(f"Error checking pod status: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking pod status: {str(e)}")
            return False

    @asynccontextmanager
    async def stream_logs(self, tail_lines=None, follow=False):
        """Context manager for log streaming to ensure proper resource cleanup"""
        log_stream = None
        try:
            log_stream = core_v1_api.read_namespaced_pod_log(
                name=self.pod_name,
                namespace=settings.NAMESPACE,
                container=self.container_name,
                follow=follow,
                _preload_content=False,
                tail_lines=tail_lines,
            )
            yield log_stream
        finally:
            if log_stream:
                try:
                    log_stream.close()
                except Exception as e:
                    logger.warning(f"Error closing log stream: {str(e)}")

    async def stream_previous_logs(self):
        """Stream previous logs in chunks to prevent memory issues"""
        try:
            # For previous logs, we use the _preload_content=True option
            # but process in smaller chunks to avoid memory issues
            previous_logs = core_v1_api.read_namespaced_pod_log(
                name=self.pod_name,
                namespace=settings.NAMESPACE,
                container=self.container_name,
                follow=False,
                _preload_content=True,
            )

            if previous_logs:
                log_lines = previous_logs.splitlines()
                # Process logs in chunks
                for i in range(0, len(log_lines), self.chunk_size):
                    chunk = log_lines[i : i + self.chunk_size]
                    if chunk:
                        await self.send_message("\n".join(chunk))
                    # Small delay to prevent overwhelming the websocket
                    await asyncio.sleep(0.1)

        except ApiException as e:
            if e.status == 404:
                await self.send_message(f"Warning: Pod {self.pod_name} logs not found")
            else:
                logger.error(f"Error fetching previous logs: {str(e)}")
                await self.send_message(f"Error fetching previous logs: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in previous logs: {str(e)}")
            await self.send_message(f"Error processing logs: {str(e)}")

    async def stream_live_logs(self):
        """Stream live logs with improved error handling and resource management"""
        # Determine how many lines to fetch
        tail_lines = None if self.full_log else self.last_lines

        try:
            async with self.stream_logs(
                tail_lines=tail_lines, follow=self.follow_log
            ) as log_stream:
                while self.is_connected:
                    # Check if pod is still running before each read
                    is_running = await self.is_pod_running(
                        self.pod_name, settings.NAMESPACE
                    )
                    if not is_running and self.follow_log:
                        await self.send_message("Info: Pod has completed execution")
                        break

                    try:
                        # Use asyncio.to_thread for non-blocking I/O
                        line = await asyncio.to_thread(log_stream.readline)

                        # Empty line might mean end of logs or a delay
                        if not line:
                            if not is_running:
                                break  # Pod finished and no more logs
                            await asyncio.sleep(1)  # Avoid busy-waiting
                            continue

                        decoded_line = line.decode("utf-8").strip()

                        # Apply search string filtering if specified
                        if self.search_string:
                            if (
                                not self.epoch_found
                                and self.search_string in decoded_line
                            ):
                                self.epoch_found = True

                            # Send all lines after we've found the first search_string
                            if self.epoch_found:
                                await self.send_message(decoded_line)
                        else:
                            await self.send_message(decoded_line)

                    except WebSocketDisconnect:
                        logger.info(
                            f"WebSocket disconnected during log streaming for job {self.job_id}"
                        )
                        break

        except ApiException as e:
            if e.status == 404:
                await self.send_message(f"Warning: Pod {self.pod_name} not found")
            else:
                logger.error(f"API error streaming logs: {str(e)}")
                await self.send_message(f"Error streaming logs: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in live logs: {str(e)}")
            await self.send_message(f"Error streaming logs: {str(e)}")

    async def run(self):
        """Main execution flow with improved error handling"""
        try:
            if not await self.wait_for_job_start():
                logger.info(f"Job {self.job_id} failed to start properly")
                return

            try:
                self.pod_name = self.get_pod_name()
                logger.info(f"Starting log stream for pod: {self.pod_name}")

                if self.full_log:
                    await self.stream_previous_logs()

                await self.stream_live_logs()

            except ValueError as e:
                await self.send_message(f"Error: {str(e)}")
                logger.error(f"Value error in log stream: {str(e)}")
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for job {self.job_id}")
            except Exception as e:
                error_msg = f"Error streaming logs: {str(e)}"
                logger.error(error_msg)
                await self.send_message(f"Error: {error_msg}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for job {self.job_id}")
        except Exception as e:
            logger.error(f"Unhandled exception in LogStreamManager: {str(e)}")
        finally:
            try:
                if self.is_connected:
                    await self.websocket.close(code=1000)
                    logger.info(f"WebSocket closed for job {self.job_id}")
            except Exception as e:
                logger.debug(f"Error closing websocket: {str(e)}")
