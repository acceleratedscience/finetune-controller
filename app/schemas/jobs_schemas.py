from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator, HttpUrl

from fastapi import UploadFile
from app.models.base.finetuning import BaseFineTuneModel
from app.core.device_config import device_configuration


class DatasetInput(BaseModel):
    dataset_id: str | None = None
    dataset_url: HttpUrl | None = None
    dataset_file: UploadFile | None = None
    dataset_description: str = ""


class JobInput(BaseModel):
    user_id: str
    job_name: str
    model_name: str
    model: BaseFineTuneModel  # type: ignore
    device: str  # type: ignore
    arguments: dict[str, Any] | None
    s3_uri: str = ""  # When dataset was uploaded as a file
    s3_artifacts_uri: str = ""
    dataset_url: str = ""  # When dataset was provided as a URL
    job_id: str = ""

    @field_validator("device")
    def validate_device(cls, name):
        if name not in device_configuration.list_workers():
            raise ValueError(
                f"Invalid device '{name}'. Must be one of {device_configuration.list_workers()}."
            )
        return name


# ------------------------
# Frontend Models
# ------------------------


# -- Jobs --


class JobMetaData(BaseModel):
    """Metadata for a PyTorch job"""

    # Job specific
    job_id: str = None
    model_name: str = None
    promotion_path: str = None
    dataset_name: str | None = None

    # PyTorch specific
    device: str = None
    task: str = None
    framework: str = None
    arguments: dict | str | None = None


class JobMeta(BaseModel):
    """Metadata for the jobs table"""

    error: str | None = None
    note: str | None = None
    data: JobMetaData = JobMetaData()


class Job(BaseModel):
    """Finetuning job"""

    index_: int  # Index used by table in frontend
    id: str
    name: str
    status: str
    status_merged: str
    promoted: str
    model_name: str
    queue_pos: int | None  # Not in UI bc behavior unpredictable
    start_time: datetime | None
    end_time: datetime | None
    duration: int | None
    promotion_path: str | None
    meta_: JobMeta = JobMeta()


# -- Dataset --


class DatasetMeta(BaseModel):
    """Metadata for the jobs table"""

    error: str | None = None
    note: str | None = None
    data: dict = {}  # s3_uri / http_url


class Dataset(BaseModel):
    """Dataset model"""

    index_: int  # Index used by table in frontend
    id: str
    name: str
    created_at: datetime
    meta_: DatasetMeta = DatasetMeta()


# -- General --


class PaginatedTableResponse(BaseModel):
    """API Response for jobs overview"""

    total: int
    totalPages: int
    resultIndices: list[int]
    page: int
    pageSize: int
    items: list[Job | Dataset]
