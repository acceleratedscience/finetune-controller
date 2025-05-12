import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from kubernetes.client.rest import ApiException
from tenacity import retry, stop_after_attempt, wait_exponential

# Assuming these imports are correctly set up in your project structure
from app.core.config import settings
from app.database.db import db_manager
from app.schemas.db_schemas import DatabaseStatusEnum, JobStatus
from app.utils.kf_config import kubeflow_api
from app.utils.kube_config import core_v1_api

logger = logging.getLogger(__name__)


class LogStreamManager:
    def __init__(
        self,
        websocket: WebSocket,
        job_id: str,
        full_log: bool,
        follow: bool,
        last_lines: int = 100,
        search_string: str | None = None,  # Allow None for no filtering
    ):
        self.websocket = websocket
        self.job_id = job_id
        self.full_log = full_log
        self.follow_log = follow
        # If full_log is requested, last_lines becomes irrelevant for the initial fetch
        self.last_lines = last_lines if not full_log else 0
        self.pod_name: str | None = None
        self.container_name = "pytorch"  # Or make configurable if needed
        self.chunk_size = 100  # Send logs in smaller chunks over websocket
        self.search_string = search_string
        self.search_string_found = not bool(
            search_string
        )  # Start sending immediately if no search string
        self.is_connected = True
        self._stream = None  # To hold the stream resource for cleanup

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def _get_job_status(self) -> JobStatus | None:
        """Fetch job status with retry mechanism"""
        return await db_manager.get_job(self.job_id)

    async def _wait_for_job_start(self) -> bool:
        """Wait for job to start with timeout"""
        MAX_WAIT_TIME = 300  # 5 minutes timeout
        start_time = asyncio.get_event_loop().time()

        while self.is_connected:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > MAX_WAIT_TIME:
                await self._send_message("Error: Timeout waiting for job to start")
                return False

            try:
                job_info = await self._get_job_status()

                if not job_info:
                    await self._send_message("Error: Job not found")
                    return False

                # Check terminal states first
                if job_info.status in [
                    DatabaseStatusEnum.failed,
                    DatabaseStatusEnum.completed,
                    DatabaseStatusEnum.error,
                ]:
                    await self._send_message(
                        f"Info: Job has already finished with status: {job_info.status.value}"
                    )
                    # Allow proceeding to fetch logs even if finished, but don't wait further.
                    return True  # Proceed to attempt log fetching

                if job_info.status == DatabaseStatusEnum.running:
                    return True  # Job is running, proceed

                # If pending or other non-running, non-terminal state
                await self._send_message(
                    f"Info: Waiting for job to start (current status: {job_info.status.value})..."
                )
                await asyncio.sleep(10)

            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket disconnected while waiting for job {self.job_id} to start"
                )
                self.is_connected = False
                return False
            except Exception as e:
                logger.error(
                    f"Error while waiting for job {self.job_id} to start: {str(e)}",
                    exc_info=True,
                )
                await self._send_message(
                    f"Error while waiting for job to start: {str(e)}"
                )
                # Depending on the error, might want to return False or retry
                return False  # Exit loop on error

        return False  # Exited loop due to disconnection

    async def _send_message(self, message: str):
        """Safe message sending with error handling"""
        if not self.is_connected:
            # Don't raise an error, just log and return, as the connection is already known to be closed.
            logger.debug(
                f"Attempted to send message on closed websocket for job {self.job_id}"
            )
            return

        try:
            await self.websocket.send_text(message)
        except (
            WebSocketDisconnect,
            RuntimeError,
        ) as e:  # Catch RuntimeError for sending on closed connection
            logger.info(
                f"WebSocket disconnected during send for job {self.job_id}: {str(e)}"
            )
            self.is_connected = False
        except Exception as e:
            logger.error(
                f"Unexpected error sending message for job {self.job_id}: {str(e)}",
                exc_info=True,
            )
            self.is_connected = False
            # Optionally re-raise or handle differently if needed

    def _get_pod_name(self) -> str:
        """Get pod name with validation"""
        try:
            # Assuming kubeflow_api correctly returns the master pod name
            pod_names = kubeflow_api.get_job_pod_names(
                self.job_id, namespace=settings.NAMESPACE, is_master=True
            )

            # Handle both list and string return types defensively
            if isinstance(pod_names, list):
                if not pod_names:
                    raise ValueError(f"No pod names found for job {self.job_id}")
                pod_name = pod_names[0]
            elif isinstance(pod_names, str) and pod_names:
                pod_name = pod_names
            else:
                raise ValueError(
                    f"Unexpected result type or empty pod name list for job {self.job_id}"
                )

            if not pod_name:
                raise ValueError(f"Empty pod name received for job {self.job_id}")

            logger.info(f"Found pod name for job {self.job_id}: {pod_name}")
            return pod_name

        except Exception as e:
            logger.error(
                f"Error getting pod name for job {self.job_id}: {str(e)}", exc_info=True
            )
            # Re-raise as ValueError to be caught in the main run method
            raise ValueError(f"Failed to get pod name: {str(e)}") from e

    async def _is_pod_running_or_pending(self) -> bool:
        """Check if the pod is still in a state where logs might be produced."""
        if not self.pod_name:
            return False
        try:
            pod = await asyncio.to_thread(
                core_v1_api.read_namespaced_pod_status,
                self.pod_name,
                settings.NAMESPACE,
            )
            # Consider Pending as potentially producing logs soon
            # Succeeded and Failed mean no more logs will be produced
            return pod.status.phase in ["Running", "Pending"]
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {self.pod_name} not found when checking status.")
                return False  # Pod is gone
            logger.warning(
                f"API Error checking pod status for {self.pod_name}: {e.status} {e.reason}"
            )
            # Uncertain state, potentially retry or assume it might still be running briefly
            return True  # Conservatively assume it might still produce logs
        except Exception as e:
            logger.error(
                f"Unexpected error checking pod status for {self.pod_name}: {str(e)}",
                exc_info=True,
            )
            return False  # Treat unexpected errors as potentially finished

    async def _stream_log_content(self):
        """Handles fetching historical and/or streaming live logs."""
        try:
            # --- Phase 1: Fetch Historical Logs ---
            if self.full_log:
                await self._send_message("Info: Fetching all previous logs...")
                try:
                    historical_logs = await asyncio.to_thread(
                        core_v1_api.read_namespaced_pod_log,
                        name=self.pod_name,
                        namespace=settings.NAMESPACE,
                        container=self.container_name,
                        follow=False,
                        _preload_content=True,  # Fetch all at once
                        timestamps=False,  # Optional: add timestamps if needed
                    )

                    if historical_logs and self.is_connected:
                        log_lines = historical_logs.splitlines()
                        for i in range(0, len(log_lines), self.chunk_size):
                            if not self.is_connected:
                                break  # Check connection before sending chunk
                            chunk = log_lines[i : i + self.chunk_size]
                            await self._process_and_send_lines(chunk)
                            await asyncio.sleep(0.05)  # Small delay to allow sending

                except ApiException as e:
                    if e.status == 404:
                        await self._send_message(
                            f"Warning: Pod {self.pod_name} logs not found for historical fetch."
                        )
                    else:
                        logger.error(
                            f"API Error fetching historical logs for {self.pod_name}: {str(e)}"
                        )
                        await self._send_message(
                            f"Error fetching historical logs: {e.reason}"
                        )
                except WebSocketDisconnect:
                    logger.info(
                        f"WebSocket disconnected during historical log fetch for {self.job_id}."
                    )
                    return  # Stop processing if disconnected
                except Exception as e:
                    logger.error(
                        f"Unexpected error fetching historical logs for {self.pod_name}: {str(e)}",
                        exc_info=True,
                    )
                    await self._send_message(
                        f"Error processing historical logs: {str(e)}"
                    )

            # Check connection before proceeding to live logs
            if not self.is_connected:
                logger.info(
                    f"WebSocket disconnected before live streaming for job {self.job_id}."
                )
                return

            # --- Phase 2: Stream Live Logs (or last N lines + Live) ---
            if self.follow_log:
                tail_value = (
                    self.last_lines
                    if not self.full_log and self.last_lines > 0
                    else None
                )
                if tail_value:
                    await self._send_message(
                        f"Info: Fetching last {tail_value} lines and following live logs..."
                    )
                else:
                    await self._send_message("Info: Following live logs...")

                try:
                    self._stream = await asyncio.to_thread(
                        core_v1_api.read_namespaced_pod_log,
                        name=self.pod_name,
                        namespace=settings.NAMESPACE,
                        container=self.container_name,
                        follow=True,
                        _preload_content=False,  # Stream content
                        tail_lines=tail_value,
                        timestamps=False,  # Optional
                        _request_timeout=None,  # Keep connection open indefinitely for follow
                    )

                    while self.is_connected:
                        try:
                            # Use asyncio.to_thread for the blocking readline() call
                            line = await asyncio.to_thread(self._stream.readline)

                            if not line:
                                # Empty response. Check if pod is still running.
                                # If pod finished, break gracefully. Otherwise, wait briefly.
                                pod_still_active = (
                                    await self._is_pod_running_or_pending()
                                )
                                if not pod_still_active:
                                    await self._send_message(
                                        "Info: Pod has likely completed execution. Stopping log follow."
                                    )
                                    break
                                else:
                                    # Pod is still active, but no logs currently. Avoid busy-loop.
                                    await asyncio.sleep(0.5)
                                    continue

                            decoded_line = line.decode(
                                "utf-8", errors="replace"
                            ).rstrip("\n")
                            await self._process_and_send_lines(
                                [decoded_line]
                            )  # Process line by line

                        except (StopIteration, asyncio.CancelledError):
                            logger.info(
                                f"Log stream ended or task cancelled for {self.pod_name}."
                            )
                            break  # Stream finished or task cancelled
                        except WebSocketDisconnect:
                            logger.info(
                                f"WebSocket disconnected during live log streaming for {self.job_id}."
                            )
                            break  # Exit loop on disconnect
                        except Exception as e:
                            # Catch potential errors during readline or decoding
                            logger.error(
                                f"Error reading from log stream for {self.pod_name}: {str(e)}",
                                exc_info=True,
                            )
                            # Check pod status; if it's gone, maybe break, otherwise continue trying?
                            pod_still_active = await self._is_pod_running_or_pending()
                            if not pod_still_active:
                                await self._send_message(
                                    "Info: Pod seems to have exited after a stream error. Stopping log follow."
                                )
                                break
                            else:
                                await self._send_message(
                                    f"Warning: Error reading log stream, will retry: {str(e)}"
                                )
                                await asyncio.sleep(1)  # Wait before retrying read

                except ApiException as e:
                    if e.status == 404:
                        await self._send_message(
                            f"Warning: Pod {self.pod_name} not found when starting live stream."
                        )
                    else:
                        logger.error(
                            f"API error starting live log stream for {self.pod_name}: {str(e)}"
                        )
                        await self._send_message(
                            f"Error starting live log stream: {e.reason}"
                        )
                except WebSocketDisconnect:
                    logger.info(
                        f"WebSocket disconnected before starting live stream for job {self.job_id}."
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error in live log streaming setup for {self.pod_name}: {str(e)}",
                        exc_info=True,
                    )
                    await self._send_message(
                        f"Error setting up live log stream: {str(e)}"
                    )

            elif not self.full_log and self.last_lines > 0:
                # Case: User wants *only* the last N lines, no full log, no follow
                await self._send_message(
                    f"Info: Fetching last {self.last_lines} lines..."
                )
                try:
                    last_n_logs = await asyncio.to_thread(
                        core_v1_api.read_namespaced_pod_log,
                        name=self.pod_name,
                        namespace=settings.NAMESPACE,
                        container=self.container_name,
                        follow=False,
                        _preload_content=True,
                        tail_lines=self.last_lines,
                    )
                    if last_n_logs and self.is_connected:
                        log_lines = last_n_logs.splitlines()
                        await self._process_and_send_lines(log_lines)

                except ApiException as e:
                    await self._send_message(
                        f"Error fetching last {self.last_lines} lines: {e.reason}"
                    )
                except WebSocketDisconnect:
                    logger.info(
                        f"WebSocket disconnected fetching last lines for job {self.job_id}."
                    )
                except Exception as e:
                    await self._send_message(
                        f"Error fetching last {self.last_lines} lines: {str(e)}"
                    )

        finally:
            # Ensure the stream resource is closed if it was opened
            self._close_stream()

    async def _process_and_send_lines(self, lines: list[str]):
        """Helper to process lines based on search string and send."""
        if not self.is_connected:
            return

        lines_to_send = []
        for line in lines:
            if not self.search_string_found and self.search_string:
                if self.search_string in line:
                    self.search_string_found = True
                    # Include the line that contained the search string
                    lines_to_send.append(line)
            elif self.search_string_found:
                # If search string was already found, or if no search string was specified
                lines_to_send.append(line)

        if lines_to_send:
            try:
                # Send lines individually or in small chunks to avoid large websocket messages
                await self._send_message("\n".join(lines_to_send))
            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket disconnected during _process_and_send_lines for job {self.job_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error sending processed lines for job {self.job_id}: {str(e)}",
                    exc_info=True,
                )
                self.is_connected = False  # Assume connection is broken on send error

    def _close_stream(self):
        """Safely close the kubernetes log stream."""
        if self._stream:
            try:
                self._stream.release_conn()  # More reliable way to close urllib3 stream
                self._stream.close()
                logger.debug(f"Closed log stream for pod {self.pod_name}")
            except Exception as e:
                logger.warning(
                    f"Error closing log stream for pod {self.pod_name}: {str(e)}"
                )
            finally:
                self._stream = None

    async def run(self):
        """Main execution flow"""
        try:
            self.is_connected = True
            logger.info(f"WebSocket accepted for job {self.job_id}")

            if not await self._wait_for_job_start():
                # Wait function already sent messages about timeout or job not found
                logger.warning(
                    f"Job {self.job_id} did not reach a running state or timed out."
                )
                # Attempt to fetch logs anyway if the job finished/failed? Decide based on requirements.
                # If we proceed, we still need the pod name.
                # Let's try to get the pod name even if not running, it might have completed/failed.
                # return # Option: Exit immediately if job didn't start

            try:
                self.pod_name = self._get_pod_name()
            except ValueError as e:
                # Error getting pod name (logged in _get_pod_name)
                await self._send_message(f"Error: {str(e)}")
                return  # Cannot proceed without pod name
            except Exception as e:  # Catch any other unexpected error
                logger.error(
                    f"Unexpected error getting pod name for job {self.job_id}: {str(e)}",
                    exc_info=True,
                )
                await self._send_message(
                    "Internal error: Failed to identify pod for logs."
                )
                return

            # Proceed to stream logs (historical and/or live)
            await self._stream_log_content()

            await self._send_message("Info: Log streaming finished.")
            logger.info(
                f"Log streaming finished for job {self.job_id}, pod {self.pod_name}"
            )

        except WebSocketDisconnect:
            logger.info(
                f"WebSocket disconnected for job {self.job_id} (caught in run)."
            )
            self.is_connected = False  # Ensure flag is set
        except Exception as e:
            logger.error(
                f"Unhandled exception in LogStreamManager run for job {self.job_id}: {str(e)}",
                exc_info=True,
            )
            # Try to send a final error message if possible
            await self._send_message(
                f"Error: An unexpected error occurred during log streaming: {str(e)}"
            )
        finally:
            self._close_stream()  # Ensure stream is closed
            if self.is_connected:
                try:
                    await self.websocket.close(code=1000)
                    logger.info(f"WebSocket closed normally for job {self.job_id}")
                except Exception as e:
                    # Log error during close but don't crash
                    logger.debug(
                        f"Error closing websocket for job {self.job_id}: {str(e)}"
                    )
            self.is_connected = False  # Mark as disconnected
