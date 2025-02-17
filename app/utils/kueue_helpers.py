import logging
from app.utils.kube_config import api_instance
from kubernetes.client.rest import ApiException
from app.core.config import settings
from app.utils.kf_config import kubeflow_api
from app.schemas.kubeflow_schemas import KubeflowStatusEnum


# Create a logger
logger = logging.getLogger(__name__)


# Define the group, version, and namespace (if applicable)
group = "kueue.x-k8s.io"
version = "v1beta1"
plural = "workloads"


async def get_kueue_queue() -> list:
    try:
        # Fetch all workloads in the namespace, sorted by creation timestamp
        workloads = api_instance.list_namespaced_custom_object(
            group=group, version=version, namespace=settings.NAMESPACE, plural=plural
        )
        # Parse workloads and extract pending jobs with their timestamps
        pending_jobs = []
        for workload in workloads.get("items", []):
            job_name = workload["metadata"].get("ownerReferences")[0].get("name")
            creation_time = workload["metadata"].get("creationTimestamp", "")
            if job_name and creation_time:
                is_pending = any(
                    condition["type"] == "QuotaReserved"
                    and condition["status"] == "False"
                    for condition in workload.get("status", {}).get("conditions", [])
                )

                if is_pending:
                    pending_jobs.append((job_name, creation_time))

        # Sort jobs by creationTimestamp (oldest first)
        pending_jobs.sort(key=lambda x: x[1])
        return [job[0] for job in pending_jobs]
    except ApiException as e:
        logger.debug(str(e))

    return []  # Job not found or not pending


async def get_kueue_position(job_id):
    try:
        # Fetch all workloads in the namespace, sorted by creation timestamp
        workloads = api_instance.list_namespaced_custom_object(
            group=group, version=version, namespace=settings.NAMESPACE, plural=plural
        )
        # Parse workloads and extract pending jobs with their timestamps
        pending_jobs = []
        for workload in workloads.get("items", []):
            job_name = workload["metadata"].get("ownerReferences")[0].get("name")
            creation_time = workload["metadata"].get("creationTimestamp", "")
            if job_name and creation_time:
                is_pending = any(
                    condition["type"] == "QuotaReserved"
                    and condition["status"] == "False"
                    for condition in workload.get("status", {}).get("conditions", [])
                )

                if is_pending:
                    pending_jobs.append((job_name, creation_time))

        # Sort jobs by creationTimestamp (oldest first)
        pending_jobs.sort(key=lambda x: x[1])
        # pending_jobs.sort(key=lambda x: datetime.fromisoformat(x[1].replace("Z", "+00:00")))

        # Find the position of the requested job_id
        for index, (queued_job_id, _) in enumerate(pending_jobs):
            if job_id in queued_job_id:
                return index + 1  # 1-based position
    except ApiException as e:
        logger.debug(str(e))

    return None  # Job not found or not pending


async def get_kubeflow_queue():
    # Fetch all jobs in the namespace
    try:
        jobs = kubeflow_api.list_jobs(namespace=settings.NAMESPACE)
        # Filter out running jobs and sort by creation timestamp
        completed_jobs = [
            job
            for job in jobs
            if job.status.conditions[-1].type.lower() in [KubeflowStatusEnum.suspended]
        ]
        # Find your job and determine its position
        queue = sorted(completed_jobs, key=lambda job: job.metadata.creation_timestamp)
        return [job.metadata.name for job in queue]
    except ApiException as e:
        logger.debug(
            f"Exception when calling CustomObjectsApi->list_namespaced_custom_object: {e}"
        )


async def get_training_job_position(job_name):
    # Fetch all jobs in the namespace
    try:
        jobs = kubeflow_api.list_jobs(namespace=settings.NAMESPACE)
        # Filter out running jobs and sort by creation timestamp
        completed_jobs = [
            job
            for job in jobs
            if job.status.conditions[-1].type.lower() in [KubeflowStatusEnum.suspended]
        ]
        # Find your job and determine its position
        queue = sorted(completed_jobs, key=lambda job: job.metadata.creation_timestamp)
        for index, job in enumerate(queue):
            if job.metadata.name == job_name:
                return index + 1  # Returning the position as 1-based index

    except ApiException as e:
        logger.debug(
            f"Exception when calling CustomObjectsApi->list_namespaced_custom_object: {e}"
        )
