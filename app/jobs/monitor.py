import asyncio
import logging
import traceback

from kubeflow.training import KubeflowOrgV1JobCondition, KubeflowOrgV1JobStatus
from kubernetes.client.rest import ApiException

from app.core.config import settings
from app.database.db import db_manager
from app.schemas.kubeflow_schemas import KubeflowStatusEnum, TrainingJobStatus
from app.utils.kf_config import kubeflow_api
from app.utils.S3Handler import s3_handler
from app.utils.kueue_helpers import get_kueue_queue, get_kubeflow_queue


# Create a logger
logger = logging.getLogger(__name__)

# Global set to track active jobs
active_jobs: set[str] = set()
# Global set to track failed jobs
failed_jobs: set[str] = set()
# Global set to track ignored jobs
ignore_jobs: set[str] = set()


class JobMonitor:
    def __init__(self):
        self.stop_monitoring = False
        self.monitoring_task = None

    async def monitor_jobs(self):
        """Background task to monitor PyTorchJobs and update MongoDB"""
        logger.debug(f"tracking jobs in namespace {settings.NAMESPACE}")
        try:
            while not self.stop_monitoring:
                try:
                    # Get all jobs from the API
                    api_response = kubeflow_api.list_jobs(namespace=settings.NAMESPACE)
                    try:
                        # if Kueue implemnted
                        queued_jobs_list = await get_kueue_queue()
                    except Exception as e:
                        # fallback to kubeflow basic queue
                        queued_jobs_list = await get_kubeflow_queue()

                    for job in api_response:
                        job_id: str = job.metadata.name
                        if job_id in ignore_jobs:
                            continue
                        if not job.status.conditions:
                            logger.warning(
                                f"job conditions not ready, skipping: {job_id}"
                            )
                            logger.debug(job.status)
                            continue
                        job_conditions: KubeflowOrgV1JobCondition = (
                            job.status.conditions[-1]
                        )  # latest condition
                        job_status: str = job_conditions.type  # the job status
                        job_lifecycle: KubeflowOrgV1JobStatus = job.status

                        # logger.debug(f"==> {job_id=} | {job_status=}")
                        if (
                            job_status not in TrainingJobStatus.stopped_states
                            and job_id not in active_jobs
                        ):
                            logger.debug(
                                f"Found job ({job_id}, status={job_status}). Adding to active jobs."
                            )
                            active_jobs.add(job_id)
                        if (
                            job_status in TrainingJobStatus.stopped_states
                            and job_id not in active_jobs
                        ):
                            #  query database to check current state
                            job_info = await db_manager.get_job(job_id)
                            if job_info and job_info.status != job_status:
                                # update needed
                                logger.debug(
                                    f"Found job ({job_id}, status={job_status}), but was not updated in db. updating..."
                                )
                                active_jobs.add(job_id)
                            elif not job_info:
                                logger.debug(
                                    f"Found Job ({job_id}, status={job_status}) but is not in db. ignoring.."
                                )
                                ignore_jobs.add(job_id)
                            else:
                                logger.debug(
                                    f"Found Job ({job_id}, status={job_status}). Already up to date. ignoring job.."
                                )
                                ignore_jobs.add(job_id)

                        # Skip if job is not being tracked
                        if job_id not in active_jobs:
                            continue

                        # logger.debug(f"updating job {job_id}:{job_status}")

                        # check queue suspended jobs if pod was created. update status state manually
                        if (
                            job_status == KubeflowStatusEnum.suspended
                            and job_lifecycle.start_time
                        ):
                            # change status to reflect pending resource in db
                            job_status = KubeflowStatusEnum.created.value

                        # get job queue position
                        queued_pos = None
                        if queued_jobs_list:
                            try:
                                queued_pos = queued_jobs_list.index(job_id) + 1
                            except ValueError:
                                # ignore, job is running
                                pass

                        # Update MongoDB with status and pod metrics
                        await db_manager.update_job_status(
                            job_id=job_id,
                            status=TrainingJobStatus.map_status(job_status),
                            metadata={
                                "last_transition_time": job_conditions.last_transition_time,
                                "last_update_time": job_conditions.last_update_time,
                                "start_time": job_lifecycle.start_time,
                                "completion_time": job_lifecycle.completion_time,
                                "message": job_conditions.message,
                                "reason": job_conditions.reason,
                                "queue_pos": queued_pos,
                                # "train_metrics": None,  # Will be updated later if available
                            },
                        )

                        # Remove completed or failed jobs from monitoring
                        if job_status in TrainingJobStatus.stopped_states:
                            active_jobs.remove(job_id)
                            ignore_jobs.add(job_id)
                            # Get final metrics from S3 if available
                            try:
                                # get databse job info
                                job_info = await db_manager.get_job(job_id)
                                if job_info:
                                    metrics = await s3_handler.get_metrics(
                                        job_info.user_id, job_id
                                    )
                                    if metrics:
                                        await db_manager.update_job_status(
                                            job_id=job_id,
                                            status=TrainingJobStatus.map_status(
                                                job_status
                                            ),
                                            metadata={
                                                "training_duration": (
                                                    job_lifecycle.completion_time
                                                    - job_lifecycle.start_time
                                                ).total_seconds(),
                                                # "train_metrics": metrics,
                                            },
                                        )
                                        # add metrics to database
                                        await db_manager.create_job_metrics(
                                            user_id=job_info.user_id,
                                            job_id=job_id,
                                            job_name=job_info.job_name,
                                            data=metrics,
                                        )
                            except Exception as e:
                                logger.debug(
                                    f"Error getting final metrics for {job_id}: {str(e)}"
                                )
                        # cleanup tasks
                        if job_status == KubeflowStatusEnum.succeeded:
                            logger.debug(f"Job {job_id} status: {job_status}")
                            await self.delete_job(job_id)
                        elif job_status == KubeflowStatusEnum.failed:
                            logger.error(
                                f"Job {job_id} failed: {job_conditions.message}"
                            )
                            # INFO: job failed, need to investigate and cleanup the job manually

                except Exception as e:
                    logger.error(f"Error in job monitoring: {str(e)}")
                    traceback.print_exc()
                    logger.debug("Stopping Job Monitor")
                    self.stop_monitoring = True
                    # Add a small delay to prevent tight error loops
                    await asyncio.sleep(5)

                # Wait before next polling cycle
                try:
                    await asyncio.sleep(settings.JOB_MONITOR_INTERVAL)
                except asyncio.CancelledError:
                    logger.info("Monitoring task cancelled during sleep")
                    break
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        finally:
            # Clean up any remaining resources
            active_jobs.clear()
            failed_jobs.clear()
            ignore_jobs.clear()

    async def delete_job(self, job_id: str):
        # delete the job from k8s
        try:
            kubeflow_api.delete_job(job_id)
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"job {job_id} not found. could not delete")
            else:
                raise

    async def start(self):
        """Start the monitoring task"""
        logger.info("starting job monitor")
        self.stop_monitoring = False
        self.monitoring_task = asyncio.create_task(
            self.monitor_jobs(), name="Job Monitor"
        )

    async def stop(self):
        """Stop the monitoring task"""
        self.stop_monitoring = True
        if self.monitoring_task:
            try:
                self.monitoring_task.cancel()
                await asyncio.shield(self.monitoring_task)
            except (asyncio.CancelledError, Exception) as e:
                logger.warning(f"Monitoring task shutdown: {str(e)}")
            finally:
                active_jobs.clear()
                failed_jobs.clear()
                ignore_jobs.clear()
                self.monitoring_task = None

    @staticmethod
    def add_job(job_id: str):
        """Add a job to active monitoring"""
        logger.debug(f"Adding job {job_id} to monitoring task")
        active_jobs.add(job_id)

    @staticmethod
    def remove_job(job_id: str):
        """Remove a job from active monitoring"""
        logger.debug(f"Removing job {job_id} from monitoring task")
        active_jobs.discard(job_id)


job_monitor = JobMonitor()
