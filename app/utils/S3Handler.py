import asyncio
import io
import logging
import os
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

import aioboto3
import pandas as pd
from botocore.exceptions import ClientError

from app.core.config import settings

# Create a logger
logger = logging.getLogger(__name__)


class S3Handler:
    def __init__(self):
        self.base_path = "finetune_jobs"
        self.session = aioboto3.Session()
        self.bucket = settings.S3_BUCKET_NAME
        if settings.AWS_ACCESS_KEY and settings.AWS_SECRET_KEY:
            self.aws_access_key = settings.AWS_ACCESS_KEY.get_secret_value()
            self.aws_secret_key = settings.AWS_SECRET_KEY.get_secret_value()
        else:
            # use system creds
            self.aws_access_key = None
            self.aws_secret_key = None
        self.region = settings.AWS_REGION

    @asynccontextmanager
    async def get_client(self):
        async with self.session.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region,
        ) as s3_client:
            yield s3_client

    def get_base_uri_path(
        self,
        user_id: str,
        job_id: str,
    ):
        return f"{self.base_path}/{user_id}/{job_id}"

    async def get_dataset_uri_string(
        self, bucket: str, user_id: str, job_id: str, path_only: bool = False
    ):
        """Generates a dataset uri string."""
        path = f"{self.get_base_uri_path(user_id, job_id)}/dataset"
        if not path_only:
            return f"s3://{bucket}/{path}"
        else:
            return path

    async def get_artifacts_uri_string(
        self, bucket: str, user_id: str, job_id: str, path_only: bool = False
    ):
        """Generates a artifacts uri string."""
        path = f"{self.get_base_uri_path(user_id, job_id)}/artifacts"
        if not path_only:
            return f"s3://{bucket}/{path}"
        else:
            return path

    async def upload_dataset(self, file_path: str, user_id: str, job_id: str) -> str:
        """Upload dataset to S3 and return the S3 URI"""
        try:
            # Extract bucket and key prefix from s3_uri
            s3_key = (
                await self.get_dataset_uri_string(self.bucket, user_id, job_id)
                + "/"
                + os.path.basename(file_path)
            )
            file_prefix = (
                await self.get_dataset_uri_string(
                    self.bucket, user_id, job_id, path_only=True
                )
                + "/"
                + os.path.basename(file_path)
            )
            logger.debug(f"uploading dataset: {file_path} -> {s3_key}")
            async with self.get_client() as s3_client:
                await s3_client.upload_file(file_path, self.bucket, file_prefix)
            return s3_key
        except ClientError as e:
            raise Exception(f"Failed to upload dataset to S3: {str(e)}")

    async def upload_dataset_bytes(
        self, data: bytes, dataset_name: str, user_id: str, job_id: str
    ) -> str:
        """Upload dataset to S3 and return the S3 URI"""
        try:
            # Extract bucket and key prefix from s3_uri
            s3_key = (
                await self.get_dataset_uri_string(self.bucket, user_id, job_id)
                + "/"
                + dataset_name
            )
            file_prefix = (
                await self.get_dataset_uri_string(
                    self.bucket, user_id, job_id, path_only=True
                )
                + "/"
                + os.path.basename(dataset_name)
            )
            logger.debug(f"uploading dataset: {dataset_name} -> {s3_key}")
            async with self.get_client() as s3_client:
                await s3_client.put_object(
                    Body=data, Bucket=self.bucket, Key=file_prefix
                )
            return s3_key
        except ClientError as e:
            raise Exception(f"Failed to upload dataset to S3: {str(e)}")

    async def stream_dataset_bytes(
        self, stream: AsyncIterator[bytes], dataset_name: str, user_id: str, job_id: str
    ) -> str:
        """Upload dataset to S3 and return the S3 URI"""
        try:
            # Extract bucket and key prefix from s3_uri
            s3_key = (
                await self.get_dataset_uri_string(self.bucket, user_id, job_id)
                + "/"
                + dataset_name
            )
            file_prefix = (
                await self.get_dataset_uri_string(
                    self.bucket, user_id, job_id, path_only=True
                )
                + "/"
                + os.path.basename(dataset_name)
            )
            logger.debug(f"streaming dataset upload: {dataset_name} -> {s3_key}")
            async with self.get_client() as s3_client:
                await s3_client.upload_fileobj(
                    stream, Bucket=self.bucket, Key=file_prefix
                )
            return s3_key
        except ClientError as e:
            raise Exception(f"Failed to upload dataset to S3: {str(e)}")

    async def validate_s3_uri(self, s3_uri: str) -> bool:
        """Validate if the S3 URI exists and is accessible"""
        try:
            if not s3_uri.startswith("s3://"):
                return False

            # Parse bucket and key from URI
            path = s3_uri.replace("s3://", "")
            bucket = path.split("/")[0]
            key = "/".join(path.split("/")[1:])

            # Check if object exists
            async with self.get_client() as s3_client:
                await s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    async def get_presigned_urls(
        self, user_id: str, job_id: str, expiration: int = 3600
    ) -> list[dict]:
        """Get presigned URLs for all artifacts in a job"""
        try:
            prefix = await self.get_artifacts_uri_string(
                self.bucket, user_id, job_id, path_only=True
            )
            async with self.get_client() as s3_client:
                objects = await s3_client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix
                )

                if "Contents" not in objects:
                    return []

                urls = []
                for obj in objects["Contents"]:
                    key = obj["Key"]
                    if not key.endswith("/"):  # Skip directories
                        url = await s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": self.bucket, "Key": key},
                            ExpiresIn=expiration,
                        )
                        urls.append({"key": os.path.basename(key), "url": url})
                return urls
        except ClientError as e:
            raise Exception(f"Failed to generate presigned URLs: {str(e)}")

    async def cleanup_job_data(self, user_id: str, job_id: str):
        """Clean up all data associated with a job"""
        try:
            prefix = await self.get_base_uri_path(user_id, job_id)
            async with self.get_client() as s3_client:
                objects = await s3_client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix
                )
                if "Contents" in objects:
                    delete_objects = {
                        "Objects": [{"Key": obj["Key"]} for obj in objects["Contents"]]
                    }
                    await s3_client.delete_objects(
                        Bucket=self.bucket, Delete=delete_objects
                    )
        except ClientError as e:
            raise Exception(f"Failed to cleanup job data: {str(e)}")

    async def cleanup_uri_items(self, s3_uri: str):
        """Clean up all data associated with a promoted job"""
        logger.debug(f"cleanup s3 items: {s3_uri}")
        try:
            # Extract bucket and key prefix from s3_uri
            _, _, bucket, *key_parts = s3_uri.split("/")
            prefix = "/".join(key_parts)  # Convert to proper key prefix

            async with self.get_client() as s3_client:
                paginator = s3_client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if "Contents" in page:
                        delete_objects = {
                            "Objects": [{"Key": obj["Key"]} for obj in page["Contents"]]
                        }
                        await s3_client.delete_objects(
                            Bucket=bucket, Delete=delete_objects
                        )
        except ClientError as e:
            raise Exception(f"Failed to cleanup job data: {str(e)}")

    async def get_metrics(self, user_id: str, job_id: str) -> list[dict]:
        """Get metrics from the metrics.json file for a specific job"""
        try:
            prefix = await self.get_artifacts_uri_string(
                self.bucket, user_id, job_id, path_only=True
            )
            async with self.get_client() as s3_client:
                objects = await s3_client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix
                )

                if "Contents" not in objects:
                    raise Exception(f"No data found for job {job_id}")

                # Find the metrics.json file
                metrics_files = [
                    obj
                    for obj in objects["Contents"]
                    # if obj["Key"].endswith("metrics.json")
                    # TODO: specify exact file per model. prone to breaking.
                    if "metrics" in obj["Key"] and obj["Key"].endswith(".csv")
                ]
                if not metrics_files:
                    raise Exception(f"No metrics file found for job {job_id}")

                # Get the most recent metrics file if multiple exist
                metrics_file = sorted(
                    metrics_files, key=lambda x: x["LastModified"], reverse=True
                )[0]

                # Get the file content
                response = await s3_client.get_object(
                    Bucket=self.bucket, Key=metrics_file["Key"]
                )
                content = await response["Body"].read()

                # # return csv data as stream
                # content = content.decode("utf-8")
                # output = io.StringIO()
                # df.to_csv(output, index=False)
                # output.seek(0)
                # return output

                # load data
                try:
                    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
                    # df = df.where(pd.notnull(df), None)
                    # Replace NaN values with None
                    df = df.fillna(0)
                    return df.to_dict(orient="records")
                except Exception as e:
                    logger.error(str(e))
                    raise ValueError("Could not read file content. corrupted?")

        except ClientError as e:
            raise Exception(f"Failed to get metrics from S3: {str(e)}")

    async def download_artifacts(self, user_id: str, job_id: str) -> tuple[str, str]:
        """Download artifacts from S3 and create a zip file"""
        temp_dir = None
        try:
            prefix = await self.get_artifacts_uri_string(
                self.bucket, user_id, job_id, path_only=True
            )
            async with self.get_client() as s3_client:
                objects = await s3_client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix
                )

                if "Contents" not in objects:
                    raise Exception(f"No artifacts found for job {job_id}")

                # Create a temporary directory to store files
                temp_dir = tempfile.mkdtemp()
                artifact_dir = os.path.join(temp_dir, "artifacts")
                os.makedirs(artifact_dir)
                logger.debug(f"creating temp dir {artifact_dir}")

                async def download_file(obj, s3_client):
                    """Helper function to download a single file"""
                    file_name = os.path.basename(obj["Key"])
                    if not file_name:  # Skip if the key ends with a directory separator
                        return
                    local_file_path = os.path.join(artifact_dir, file_name)
                    await s3_client.download_file(
                        self.bucket, obj["Key"], local_file_path
                    )
                    # logger.debug(f"Downloaded {obj['Key']} to {local_file_path}")

                # Download all artifacts files concurrently
                download_tasks = [
                    download_file(obj, s3_client) for obj in objects["Contents"]
                ]
                await asyncio.gather(*download_tasks)

                # Create zip file in a streaming way
                zip_path = os.path.join(temp_dir, f"artifacts_{job_id}.zip")

                def add_file_to_zip(zip_file, file_path, arcname):
                    """Add a file to ZIP archive in chunks"""
                    CHUNK_SIZE = 8192  # 8KB chunks
                    zip_info = zipfile.ZipInfo.from_file(file_path, arcname)
                    with zip_file.open(zip_info, mode="w") as dest:
                        with open(file_path, "rb") as f:
                            while True:
                                chunk = f.read(CHUNK_SIZE)
                                if not chunk:
                                    break
                                dest.write(chunk)

                async def create_zip():
                    with zipfile.ZipFile(
                        zip_path, "w", zipfile.ZIP_DEFLATED
                    ) as zip_file:
                        artifact_path = Path(artifact_dir)
                        for file_path in artifact_path.rglob("*"):
                            if file_path.is_file():
                                arcname = file_path.relative_to(artifact_path)
                                await asyncio.to_thread(
                                    add_file_to_zip,
                                    zip_file,
                                    str(file_path),
                                    str(arcname),
                                )
                                # logger.debug(f"Added {arcname} to ZIP archive")

                await create_zip()
                return zip_path, temp_dir

        except ClientError as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise Exception(f"Failed to download artifacts from S3: {str(e)}")
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e

    async def copy_s3_object(self, source_s3_uri: str, destination_s3_uri: str):
        """Copy an object or a folder from one S3 URI to another in a non-blocking manner."""
        try:
            if not source_s3_uri.startswith(
                "s3://"
            ) or not destination_s3_uri.startswith("s3://"):
                raise ValueError("Invalid S3 URI format")

            source_path = source_s3_uri.replace("s3://", "").split("/", 1)
            destination_path = destination_s3_uri.replace("s3://", "").split("/", 1)

            if len(source_path) < 2 or len(destination_path) < 2:
                raise ValueError("Invalid S3 URI format")

            source_bucket, source_key = source_path
            destination_bucket, destination_key = destination_path

            async with self.get_client() as s3_client:
                # Check if source object exists
                try:
                    await s3_client.head_object(Bucket=source_bucket, Key=source_key)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        # Source key doesn't exist; check if it's a folder
                        response = await s3_client.list_objects_v2(
                            Bucket=source_bucket, Prefix=source_key
                        )
                        if "Contents" not in response:
                            raise Exception(f"No objects found at {source_s3_uri}")

                        # Copy all files under the prefix
                        copy_tasks = []
                        for obj in response["Contents"]:
                            file_key = obj["Key"]
                            new_key = file_key.replace(
                                source_key.rstrip("/"), destination_key.rstrip("/"), 1
                            )

                            copy_source = {"Bucket": source_bucket, "Key": file_key}
                            copy_tasks.append(
                                s3_client.copy_object(
                                    CopySource=copy_source,
                                    Bucket=destination_bucket,
                                    Key=new_key,
                                )
                            )

                        await asyncio.gather(*copy_tasks)
                        logger.info(
                            f"Copied folder {source_s3_uri} to {destination_s3_uri}"
                        )
                        return
                    else:
                        raise Exception(f"Failed to check S3 object: {str(e)}")

                # Copy a single file
                copy_source = {"Bucket": source_bucket, "Key": source_key}
                await s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=destination_bucket,
                    Key=destination_key,
                )
                logger.info(f"Copied {source_s3_uri} to {destination_s3_uri}")
        except ClientError as e:
            raise Exception(f"Failed to copy S3 object: {str(e)}")


# Create singleton instance
s3_handler = S3Handler()
