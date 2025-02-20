import logging

from app.database.db import db_manager
from app.jobs.kubeflow.PyTorchJobDeployer import PyTorchJobDeployer
from app.schemas.jobs_schemas import JobInput, DatasetInput
from app.utils.S3Handler import s3_handler
from app.core.config import settings
from app.utils.dataset_helpers import (
    upload_dataset_file,
    stream_dataset_url,
)
from fastapi.exceptions import HTTPException


# Create a logger
logger = logging.getLogger(__name__)


async def task_builder(
    job: JobInput,
    namespace: str,
    dataset_input: DatasetInput,
):
    """Create the finetune job task and run it"""
    dataset_doc = None
    if dataset_input.dataset_id:
        # use dataset already uploaded
        db_info = await db_manager.update_dataset(
            job.user_id, dataset_input.dataset_id, job.job_id
        )
        if db_info:
            logger.debug(f"reusing dataset: {db_info.dataset.s3_uri}")
            job.s3_uri = db_info.dataset.s3_uri
            job.model.dataset_info.dataset_name = db_info.dataset_name
        else:
            raise HTTPException(
                status_code=404, detail="Selected dataset not available"
            )
    elif dataset_input.dataset_url:
        # Dataset from URL to upload
        dataset_doc = await stream_dataset_url(
            job, str(dataset_input.dataset_url), dataset_input.dataset_description
        )
        # await upload_dataset_url(job, str(dataset_input.dataset_url))
    elif dataset_input.dataset_file:
        # Dataset from file upload
        dataset_doc = await upload_dataset_file(
            job, dataset_input.dataset_file, dataset_input.dataset_description
        )

    # create artifacts assests uri (where training data is stored)
    job.s3_artifacts_uri = await s3_handler.get_artifacts_uri_string(
        settings.S3_BUCKET_NAME, job.user_id, job.job_id
    )

    # Create and deploy PyTorch job
    task = PyTorchJobDeployer(namespace=namespace)
    result = task.create_pytorch_job(job=job)

    # Create job entry in MongoDB
    await db_manager.create_job(
        user_id=job.user_id,
        job_id=job.job_id,
        job_name=job.job_name,
        model_name=job.model_name,
        device=job.device,
        task=job.model.task.value,
        framework=job.model.framework.value,
        arguments=job.arguments,
        dataset_id=dataset_doc.id if dataset_doc else None,
        atrifacts_uri=job.s3_artifacts_uri,
        dataset_name=job.model.dataset_info.dataset_name
        if job.model.dataset_info.dataset_name
        else None,
        metadata={"kubernetes_job_name": result.get("metadata", {}).get("name")},
    )

    return result
