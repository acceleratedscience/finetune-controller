import asyncio
import logging

from kubeflow.training import KubeflowOrgV1JobCondition, KubeflowOrgV1JobStatus
from kubernetes.client.rest import ApiException

from app.core.config import settings
from app.database.db import db_manager
from app.schemas.db_schemas import JobStatus
from app.schemas.kubeflow_schemas import KubeflowStatusEnum, TrainingJobStatus
from app.utils.kf_config import kubeflow_api
from app.utils.S3Handler import s3_handler
from app.utils.kueue_helpers import get_kueue_queue, get_kubeflow_queue

logger = logging.getLogger(__name__)


class JobMonitor:
    def __init__(self):
        self.stop_monitoring = False
        self.monitoring_task = None

    async def _get_queue_info(self) -> dict[str, int]:
        """Get queue information with fallback"""
        try:
            return {job: idx + 1 for idx, job in enumerate(await get_kueue_queue())}
        except Exception as e:
            logger.exception("Kueue queue fetch failed, falling back to Kubeflow queue")
            return {job: idx + 1 for idx, job in enumerate(await get_kubeflow_queue())}
        except Exception as e:
            logger.error(f"Failed to get queue information: {e}")
            return {}

    async def _process_job_metrics(
        self,
        job_id: str,
        job_info,
        job_lifecycle: KubeflowOrgV1JobStatus,
        is_completed: bool = False,
    ):
        """Process and store job metrics"""
        try:
            metrics = None
            try:
                metrics = await s3_handler.get_metrics(job_info.user_id, job_id)
            except Exception as e:
                return

            if not metrics:
                return

            update_tasks = []

            # For completed jobs, update training duration
            if (
                is_completed
                and job_lifecycle.start_time
                and job_lifecycle.completion_time
            ):
                training_duration = (
                    job_lifecycle.completion_time - job_lifecycle.start_time
                ).total_seconds()

                update_tasks.append(
                    db_manager.update_job_status(
                        job_id=job_id,
                        status=job_info.status,
                        metadata={"training_duration": training_duration},
                    )
                )

            # Check if metrics document exists and update accordingly
            existing_metrics = await db_manager.get_job_metrics(job_id)
            if existing_metrics:
                update_tasks.append(
                    db_manager.job_metrics_update(
                        user_id=job_info.user_id, job_id=job_id, data=metrics
                    )
                )
            else:
                logger.debug(f"Retrieved metrics from S3 for job {job_id}")
                update_tasks.append(
                    db_manager.create_job_metrics(
                        user_id=job_info.user_id,
                        job_id=job_id,
                        job_name=job_info.job_name,
                        data=metrics,
                    )
                )

            if update_tasks:
                await asyncio.gather(*update_tasks)

        except Exception as e:
            logger.error(f"Error processing metrics for job {job_id}: {e}")

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        conditions: KubeflowOrgV1JobCondition,
        lifecycle: KubeflowOrgV1JobStatus,
        queue_position: int | None = None,
    ) -> JobStatus | None:
        """Update job status in database"""
        try:
            return await db_manager.update_job_status(
                job_id=job_id,
                status=TrainingJobStatus.map_status(status),
                metadata={
                    "last_transition_time": conditions.last_transition_time,
                    "last_update_time": conditions.last_update_time,
                    "start_time": lifecycle.start_time,
                    "completion_time": lifecycle.completion_time,
                    "message": conditions.message,
                    "reason": conditions.reason,
                    "queue_pos": queue_position,
                },
            )
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {e}")
            raise

    async def monitor_jobs(self):
        """Background task to monitor PyTorchJobs and update MongoDB"""
        logger.info(f"Starting job monitoring in namespace {settings.NAMESPACE}")

        while not self.stop_monitoring:
            try:
                # Get current state from Kubeflow API
                jobs = kubeflow_api.list_jobs(namespace=settings.NAMESPACE)
                queue_positions = await self._get_queue_info()

                for job in jobs:
                    job_id = job.metadata.name

                    if not job.status.conditions:
                        logger.warning(f"Job conditions not ready for {job_id}")
                        await asyncio.sleep(0.1)
                        continue

                    conditions = job.status.conditions[-1]
                    status: str = conditions.type

                    # Handle suspended jobs that have actually started
                    if status == KubeflowStatusEnum.suspended and job.status.start_time:
                        status = KubeflowStatusEnum.created.value

                    # For completed jobs, check if we need to update the database
                    if status in TrainingJobStatus.stopped_states:
                        job_info = await db_manager.get_job(job_id)
                        if job_info and job_info.status == status:
                            # ignore jobs that are already in stopped states
                            await asyncio.sleep(0.1)
                            continue

                    # check if state changed
                    prev_job_info = await db_manager.get_job(job_id)
                    if prev_job_info and prev_job_info.status.lower() != status.lower():
                        logger.info(
                            f"Job {job_id} status changed from {prev_job_info.status} to {status}"
                        )

                    # Update status in database
                    job_info = await self._update_job_status(
                        job_id,
                        status,
                        conditions,
                        job.status,
                        queue_positions.get(job_id),
                    )

                    if job_info:
                        # Process metrics for both running and completed jobs
                        is_completed = status in TrainingJobStatus.stopped_states
                        await self._process_job_metrics(
                            job_id, job_info, job.status, is_completed=is_completed
                        )

                        # Additional handling for completed jobs
                        if is_completed:
                            if status == KubeflowStatusEnum.succeeded:
                                logger.info(
                                    f"Job {job_id} completed successfully, cleaning up"
                                )
                                await self.delete_job(job_id)
                            elif status == KubeflowStatusEnum.failed:
                                logger.error(
                                    f"Job {job_id} failed. Manual investigation required."
                                )

            except Exception as e:
                logger.error(f"Error in job monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(settings.JOB_MONITOR_INTERVAL)

    async def delete_job(self, job_id: str):
        """Delete a job from Kubernetes"""
        try:
            await asyncio.to_thread(kubeflow_api.delete_job, job_id)
        except ApiException as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise

    async def start(self):
        """Start the monitoring task"""
        logger.info("Starting job monitor")
        self.stop_monitoring = False
        self.monitoring_task = asyncio.create_task(self.monitor_jobs())

    async def stop(self):
        """Stop the monitoring task"""
        logger.info("Stopping job monitor")
        self.stop_monitoring = True
        if self.monitoring_task:
            try:
                self.monitoring_task.cancel()
                await asyncio.shield(self.monitoring_task)
            except asyncio.CancelledError:
                pass
            finally:
                self.monitoring_task = None


job_monitor = JobMonitor()
