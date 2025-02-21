from datetime import datetime
from enum import Enum
from bson import ObjectId
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# Custom Pydantic field to handle ObjectId
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field):
        if not isinstance(v, ObjectId):
            raise ValueError("Not a valid ObjectId")
        return str(v)


class DatasetTypes(BaseModel):
    s3_uri: str = ""
    http_url: str = ""


# Define the dataset schema
class DatasetModel(BaseModel):
    model_config = ConfigDict(
        # Fields may be added by database query pipeline:
        # - index_
        extra="allow",
        protected_namespaces=(),
        populate_by_name=True,  # Allow "_id" alias for "id"
        from_attributes=True,  # Required for ORM-like conversion
    )
    id: PyObjectId | None = Field(alias="_id")  # Converts MongoDB ObjectId to string
    user_id: str
    dataset: DatasetTypes  # Example field, add more fields as needed
    dataset_name: str
    description: str = ""
    job_ref: list[str] = []
    created_at: datetime | None = None


class DatabaseStatusEnum(str, Enum):
    """database states"""

    model_config = ConfigDict(
        # Fields may be added by database query pipeline:
        # - status_merged
        extra="allow",
        protected_namespaces=(),
    )

    # running states
    queued = "queued"
    starting = "starting"
    restarting = "restarting"
    running = "running"

    # stopped states
    completed = "completed"
    failed = "failed"
    canceled = "canceled"
    error = "error"


class PromotionStatus(str, Enum):
    NOT_PROMOTED = "not_promoted"
    IN_PROGRESS = "in_progress"
    DELETING = "deleting"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    start_time: datetime | None = None
    completion_time: datetime | None = None
    cancellation_time: datetime | None = None
    queue_pos: int | None = None


class JobStatus(BaseModel):
    model_config = ConfigDict(
        # Fields may be added by database query pipeline:
        # - index_
        # - start_time
        # - end_time
        # - duration
        # - status_merged
        extra="allow",
        protected_namespaces=(),
    )

    user_id: str
    job_id: str
    job_name: str
    status: DatabaseStatusEnum
    promoted: PromotionStatus = PromotionStatus.NOT_PROMOTED
    created_at: datetime
    updated_at: datetime
    model_name: str
    device: str
    task: str
    framework: str
    arguments: dict[str, Any] | None = None
    dataset_id: str | None = None
    atrifacts_uri: str | None = None
    destination_uri: str | None = None
    dataset_name: str | None = None
    metadata: JobStatusMetadata | None = None


class JobsPage(BaseModel):
    """One page of a user's jobs."""

    items: list[JobStatus]
    total: int
    total_pages: int


class DatasetPage(BaseModel):
    """One page of a user's datasets."""

    items: list[DatasetModel]
    total: int
    total_pages: int


class MetricsDocument(BaseModel):
    user_id: str
    job_id: str
    job_name: str
    metrics: object
