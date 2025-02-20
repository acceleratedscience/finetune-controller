import logging
import os

import aiofiles
import aiohttp
from fastapi import UploadFile
from urllib.parse import urlparse

from app.database.db import db_manager
from app.schemas.jobs_schemas import JobInput
from app.schemas.db_schemas import DatasetTypes, DatasetModel
from app.utils.S3Handler import s3_handler
from app.utils.naming import generate_short_uuid


# Create a logger
logger = logging.getLogger(__name__)


async def upload_dataset_file(
    job: JobInput, dataset_file: UploadFile, description: str
) -> DatasetModel | None:
    """Upload dataset file to s3 and update job `JobInput` object"""
    # Create temp directory if it doesn't exist
    dataset_doc = None
    temp_dir = "/tmp/ftjobs"
    os.makedirs(temp_dir, exist_ok=True)
    # Save uploaded file temporarily
    logger.debug(f"saving dataset ({dataset_file.filename}) to disk")
    temp_file_path = os.path.join(temp_dir, dataset_file.filename)
    async with aiofiles.open(temp_file_path, "wb") as f:
        content = await dataset_file.read()
        await f.write(content)
    try:
        # Upload to S3
        logger.debug(f"uploading dataset ({dataset_file.filename})")
        s3_uri = await s3_handler.upload_dataset(
            temp_file_path, job.user_id, job.job_id
        )
        # insert dataset info to db
        dataset_doc = db_info = await db_manager.insert_dataset(
            user_id=job.user_id,
            job_id=job.job_id,
            dataset=DatasetTypes(s3_uri=s3_uri),
            dataset_name=dataset_file.filename,
            description=description,
        )
        # Update job's dataset info with S3 URI
        job.s3_uri = db_info.dataset.s3_uri
        # update model dataset_name
        job.model.dataset_info.dataset_name = db_info.dataset_name
    finally:
        # Clean up temp file
        logger.debug(f"cleaning up dataset ({dataset_file.filename}) from disk")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return dataset_doc if dataset_doc else None


async def get_filename_from_response(response, url: str) -> str:
    """Extract filename from Content-Disposition header or URL path."""
    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition and "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip().strip('"')
    else:
        # Fallback to extracting from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

    return filename or "default_filename-" + generate_short_uuid()


async def download_from_url(url: str) -> bytes:
    """Download content from the given URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
            filename = await get_filename_from_response(response, url)
            return data, filename


async def upload_dataset_url(job: JobInput, url: str) -> DatasetModel | None:
    """Download dataset and Upload dataset to s3 and update job `JobInput` object"""
    dataset_doc = None
    logger.debug(f"downloading dataset from url: '{url}'")
    try:
        # download data
        data, filename = await download_from_url(url)
        s3_uri = await s3_handler.upload_dataset_bytes(
            data=data,
            dataset_name=filename,
            user_id=job.user_id,
            job_id=job.job_id,
        )
        # insert dataset info to db
        db_info = await db_manager.insert_dataset(
            user_id=job.user_id,
            job_id=job.job_id,
            dataset=DatasetTypes(s3_uri=s3_uri, http_url=url),
            dataset_name=filename,
            description="from url",
        )
        # Update job's dataset info with S3 URI
        job.s3_uri = db_info.dataset.s3_uri
        # update model dataset_name
        job.model.dataset_info.dataset_name = db_info.dataset_name
    except Exception as e:
        raise e
    return dataset_doc if dataset_doc else None


async def stream_dataset_url(
    job: JobInput, url: str, description: str
) -> DatasetModel | None:
    """Stream file from URL to S3 without loading into memory."""
    logger.debug(f"streaming upload dataset from url: '{url}'")
    dataset_doc = None
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.get(url) as response:
                response.raise_for_status()
                filename = await get_filename_from_response(response, url)
                s3_uri = await s3_handler.stream_dataset_bytes(
                    stream=response.content,
                    dataset_name=filename,
                    user_id=job.user_id,
                    job_id=job.job_id,
                )
            logger.debug("done uploading stream")
        # insert dataset info to db
        dataset_doc = db_info = await db_manager.insert_dataset(
            user_id=job.user_id,
            job_id=job.job_id,
            dataset=DatasetTypes(s3_uri=s3_uri, http_url=url),
            dataset_name=filename,
            description=description,
        )
        # Update job's dataset info with S3 URI
        job.s3_uri = db_info.dataset.s3_uri
        # update model dataset_name
        job.model.dataset_info.dataset_name = db_info.dataset_name
    except Exception as e:
        raise e
    return dataset_doc if dataset_doc else None
