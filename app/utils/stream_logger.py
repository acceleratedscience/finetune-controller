import asyncio
from kubernetes.client.rest import ApiException
from fastapi import WebSocket, WebSocketDisconnect
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from app.database.db import db_manager
from app.schemas.db_schemas import DatabaseStatusEnum
from app.schemas.kubeflow_schemas import TrainingJobStatus
from app.utils.kf_config import kubeflow_api
from app.core.config import settings
from app.utils.kube_config import core_v1_api


logger = logging.getLogger(__name__)


class LogStreamManager:
    def __init__(self, websocket: WebSocket, job_id: str, full_log: bool, follow: bool):
        self.websocket = websocket
        self.job_id = job_id
        self.full_log = full_log
        self.follow_log = follow
        self.pod_name: str | None = None
        self.container_name = "pytorch"
        self.chunk_size = 4096  # Optimal chunk size for streaming
        self.retry_attempts = 3
        self.backoff_factor = 2

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def get_job_status(self) -> dict:
        """Fetch job status with retry mechanism"""
        return await db_manager.get_job(self.job_id)

    async def wait_for_job_start(self) -> bool:
        """Wait for job to start with timeout"""
        MAX_WAIT_TIME = 300  # 5 minutes timeout
        start_time = asyncio.get_event_loop().time()
        while True:
            job_info = await self.get_job_status()

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

    async def send_message(self, message: str):
        """Safe message sending with error handling"""
        try:
            await self.websocket.send_text(message)
        except Exception as e:
            # logger.error(f"Failed to send message: {str(e)}")
            raise WebSocketDisconnect()

    def get_pod_name(self) -> str:
        """Get pod name with validation"""
        pod_names = kubeflow_api.get_job_pod_names(
            self.job_id, namespace=settings.NAMESPACE, is_master=True
        )

        if isinstance(pod_names, list) and pod_names:
            return pod_names[0]
        elif isinstance(pod_names, str):
            return pod_names
        raise ValueError(f"Invalid pod name response: {pod_names}")

    async def is_pod_running(self, pod_name: str, namespace: str):
        """Check if the pod is still running."""
        try:
            pod = core_v1_api.read_namespaced_pod_status(pod_name, namespace)
            return pod.status.phase in ["Running", "Pending"]
        except Exception as e:
            return False  # Assume pod is not running if we can't fetch status

    async def stream_previous_logs(self):
        """Stream previous logs in reverse order to maintain chronological sequence"""
        try:
            previous_logs = core_v1_api.read_namespaced_pod_log(
                name=self.pod_name,
                namespace=settings.NAMESPACE,
                container=self.container_name,
                follow=False,
                _preload_content=True,
            )

            if previous_logs:
                # Stream in chunks to prevent memory issues
                for chunk in [
                    previous_logs[i : i + self.chunk_size]
                    for i in range(0, len(previous_logs), self.chunk_size)
                ]:
                    # reversed_chunk = "\n".join(chunk.splitlines()[::-1])  # Reverse log lines
                    await self.send_message(chunk)

        except ApiException as e:
            # logger.error(f"Error fetching previous logs: {str(e)}")
            raise

    async def stream_live_logs(self):
        """Stream live logs with error handling"""
        try:
            log_stream = core_v1_api.read_namespaced_pod_log(
                name=self.pod_name,
                namespace=settings.NAMESPACE,
                container=self.container_name,
                follow=self.follow_log,
                _preload_content=False,
                tail_lines=1,
            )

            while await self.is_pod_running(self.pod_name, settings.NAMESPACE):
                line = await asyncio.to_thread(log_stream.readline)
                if not line:
                    await asyncio.sleep(1)  # Avoid busy-waiting
                    continue
                print(line)
                await self.send_message(line.decode("utf-8").strip())

        except ApiException as e:
            if e.status == 404:
                await self.send_message(f"Warning: Pod {self.pod_name} not found")
            else:
                raise

    async def _debug(self):
        while True:
            await asyncio.sleep(1)
            msg = []
            for i in range(10):
                msg.append(f"This is a debug message {asyncio.get_event_loop().time()}")
            await self.send_message("\n".join(msg))

    async def run(self):
        """Main execution flow"""
        # await self._debug()

        try:
            if not await self.wait_for_job_start():
                return

            self.pod_name = self.get_pod_name()
            logger.info(f"Starting log stream for pod: {self.pod_name}")

            if self.full_log:
                # stream previous logs
                await self.stream_previous_logs()
            await self.stream_live_logs()

        except Exception as e:
            pass  # some error in connection. ignore and close.
        finally:
            try:
                await self.websocket.close()
            except Exception as e:
                pass  # user closed connection already
