import logging
from datetime import datetime, timezone
from typing import Any
from bson import ObjectId

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings
from app.schemas.db_schemas import (
    DatabaseStatusEnum,
    JobsPage,
    DatasetPage,
    JobStatus,
    MetricsDocument,
    PromotionStatus,
    DatasetModel,
    DatasetTypes,
)

# Create a logger
logger = logging.getLogger(__name__)


class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.jobs_collection = None
        self.datasets_collection = None
        self.metrics_collection = None
        self.archived_jobs_collection = None

    async def connect(self):
        try:
            # Get MongoDB connection details from environment variables
            mongodb_url = settings.MONGODB_URL
            logger.debug(f"Initializing MongoDB connection to {mongodb_url}")

            # Setup connection options with authentication if credentials are provided
            connect_args = {}
            if settings.MONGODB_USERNAME and settings.MONGODB_PASSWORD:
                logger.debug("Using authenticated connection")
                connect_args.update(
                    {
                        "username": settings.MONGODB_USERNAME.get_secret_value(),
                        "password": settings.MONGODB_PASSWORD.get_secret_value(),
                    }
                )

            # Create authenticated connection
            self.client = AsyncIOMotorClient(mongodb_url, **connect_args)

            # Initialize database
            self.db = self.client[settings.MONGODB_DATABASE]
            logger.debug(f"Using database: {settings.MONGODB_DATABASE}")

            # Get jobs collection
            self.jobs_collection = self.db.jobs
            # Get metrics collection
            self.metrics_collection = self.db.metrics
            # Get datasets collection
            self.datasets_collection = self.db.datasets
            # Get archived jobs collection
            self.archived_jobs_collection = self.db.archived_jobs
            if self.client:
                await self._ensure_indexes()
            logger.debug("MongoDB manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB manager: {str(e)}")
            raise

    async def close(self):
        if self.client:
            self.client.close()

    async def _ensure_indexes(self):
        """Create indexes if they don't exist"""
        try:
            # Index for faster job lookups
            await self.jobs_collection.create_index("user_id")
            await self.jobs_collection.create_index("job_id", unique=True)
            await self.jobs_collection.create_index(
                [("job_name", "text"), ("model_name", "text")]
            )
            # Compound index for user+status queries
            await self.jobs_collection.create_index([("user_id", 1), ("status", 1)])
            await self.jobs_collection.create_index([("user_id", 1), ("job_id", 1)])

            # archived jobs collection
            await self.archived_jobs_collection.create_index("user_id")
            await self.archived_jobs_collection.create_index("job_id", unique=True)

            # metrics collection
            await self.metrics_collection.create_index("user_id")
            await self.metrics_collection.create_index("job_id", unique=True)
            await self.metrics_collection.create_index([("user_id", 1), ("job_id", 1)])

            # datasets collection
            await self.datasets_collection.create_index([("_id", 1), ("user_id", 1)])
            await self.datasets_collection.create_index("user_id")
            logger.debug("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    async def create_job(
        self,
        user_id: str,
        job_id: str,
        job_name: str,
        model_name: str,
        device: str,
        task: str,
        framework: str,
        arguments: dict[str, Any] | None = None,
        dataset_id: str | None = None,
        atrifacts_uri: str | None = None,
        dataset_name: str | None = None,
        metadata: dict | None = None,
    ) -> JobStatus:
        """Create a new job entry in the database."""
        now = datetime.now(tz=timezone.utc)
        job_data = {
            "user_id": user_id,
            "job_id": job_id,
            "job_name": job_name,
            "status": DatabaseStatusEnum.queued,
            "promoted": PromotionStatus.NOT_PROMOTED,
            "created_at": now,
            "updated_at": now,
            "model_name": model_name,
            "device": device,
            "task": task,
            "framework": framework,
            "arguments": arguments,
            "dataset_id": dataset_id,
            "atrifacts_uri": atrifacts_uri,
            "dataset_name": dataset_name,
            "metadata": metadata,
        }

        # create the model
        job = JobStatus(**job_data)

        # insert data into database
        await self.jobs_collection.insert_one(job.model_dump())
        return job

    async def create_job_metrics(
        self, user_id: str, job_id: str, job_name: str, data: object
    ) -> MetricsDocument:
        try:
            metrics_data = {
                "user_id": user_id,
                "job_id": job_id,
                "job_name": job_name,
                "metrics": data,
            }
            await self.metrics_collection.insert_one(metrics_data)
            return MetricsDocument(**metrics_data)
        except Exception as e:
            logger.error(f"Failed to add metrics to db: {str(e)}")
            raise

    async def job_metrics_update(
        self, user_id: str, job_id: str, data: list[dict]
    ) -> bool:
        try:
            metrics_data = await self.metrics_collection.find_one(
                {"job_id": job_id, "user_id": user_id}
            )
            if metrics_data:
                # Merge new metadata with existing metadata
                await self.metrics_collection.update_one(
                    {"job_id": job_id}, {"$set": {"metrics": data}}
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update metrics to db: {str(e)}")
            raise

    async def get_job_metrics(self, job_id: str) -> MetricsDocument:
        try:
            result = await self.metrics_collection.find_one({"job_id": job_id})
            if result:
                return MetricsDocument(**result)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get metrics from db: {str(e)}")
            raise

    async def update_job_status(
        self, job_id: str, status: str, metadata: dict | None = None
    ) -> JobStatus | None:
        """Update the status of an existing job."""
        try:
            # Always update status and timestamp
            update_data = {
                "status": status,
                "updated_at": datetime.now(tz=timezone.utc),
            }

            if metadata:
                # Get current job to merge metadata
                current_job = await self.jobs_collection.find_one({"job_id": job_id})
                if current_job:
                    # Merge new metadata with existing metadata
                    current_metadata = current_job.get("metadata", {})
                    merged_metadata = {**current_metadata, **metadata}
                    update_data["metadata"] = merged_metadata
                else:
                    update_data["metadata"] = metadata

            result = await self.jobs_collection.update_one(
                {"job_id": job_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                job = await self.get_job(job_id)
                return job
            return None

        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
            raise

    async def update_job_promotion(
        self,
        job_id: str,
        value: PromotionStatus,
        destination_uri: str | None = None,
    ) -> JobStatus | None:
        """Update the status of an existing job."""
        try:
            # Always update status and timestamp
            update_data = {
                "promoted": value,
                "destination_uri": destination_uri,
            }

            result = await self.jobs_collection.update_one(
                {"job_id": job_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                job = await self.get_job(job_id)
                return job
            return None

        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
            raise

    async def get_job(self, job_id: str) -> JobStatus | None:
        """Retrieve a job by its ID."""

        # Query pipeline
        pipeline = [{"$match": {"job_id": job_id}}]
        # Add start_time, end_time, and duration
        pipeline = pipeline + self._job_pipeline_add_fields()
        # Exclude _id field, this causes pydantic error
        pipeline.append({"$unset": "_id"})

        cursor = self.jobs_collection.aggregate(pipeline)
        job_info = await cursor.to_list(length=1)
        job_info = job_info[0] if job_info else None
        if job_info:
            return JobStatus(**job_info)
        return None

    async def get_all_user_jobs(self, user_id: str) -> list[JobStatus]:
        """Retrieve all jobs for a specific user."""
        cursor = self.jobs_collection.find({"user_id": user_id})
        jobs = []
        async for job in cursor:
            jobs.append(JobStatus(**job))
        return jobs

    async def get_user_jobs(
        self,
        user_id: str,
        page: int,
        page_size: int,
        sort: str,
        query: str,
        limit: list[int] | None,
        status: str | None,
        model_name: str | None,
    ) -> JobsPage:
        """
        Retrieve paginated, sorted jobs for a specific user.

        Prepare the data to be consumed by the frontend table component.
        """

        # Formulate query
        # ----------------

        skip = (page - 1) * page_size
        mongo_query = {"user_id": user_id}

        # Filter by status
        if status:
            # "Promoted" is treated as a status in the frontend, but it's a flag in the database
            if status == "promoted":
                # ignore status query instead query promoted flag
                mongo_query["promoted"] = PromotionStatus.COMPLETED
            else:
                mongo_query["status"] = status

        # Filter by model name
        if model_name:
            mongo_query["model_name"] = model_name

        # Filter by search query
        if query:
            # mongo_query['$text'] = {"$search": query} # Faster but requires full word match
            mongo_query["job_name"] = {"$regex": query, "$options": "i"}

        # Formulate Sort
        # ----------------

        # Default sort key
        sort = sort or "-start_time"

        # Sort order
        sort_order = 1
        if sort and sort.startswith("-"):
            sort_order = -1
            sort = sort[1:]

        # Query pipeline
        # ----------------

        pipeline = [{"$match": mongo_query}]
        pipeline = (
            pipeline + self._job_pipeline_add_fields()
        )  # Add start_time, end_time, and duration
        pipeline = pipeline + [
            # Add index field, needed for the table
            {
                "$setWindowFields": {
                    "sortBy": {sort: sort_order},
                    "output": {"index_": {"$documentNumber": {}}},
                }
            },
            # Make index start from zero
            # {"$addFields": {"index_": {"$subtract": ["$index", 1]}}},
            # Sort by the requested field
            {"$sort": {sort: sort_order}},
        ]

        # This lets user limit the results to only the selected jobs
        if limit is not None:
            pipeline.append({"$match": {"index_": {"$in": limit}}})

        # Pagination
        pipeline = pipeline + [
            {"$skip": skip},
            {"$limit": page_size},
        ]

        # Fetch
        cursor = self.jobs_collection.aggregate(pipeline)

        # Assemble output
        # ----------------

        total_jobs = await self.jobs_collection.count_documents(mongo_query)
        total_pages = (total_jobs + page_size - 1) // page_size

        jobs = []
        async for job in cursor:
            jobs.append(JobStatus(**job))

        return JobsPage(items=jobs, total=total_jobs, total_pages=total_pages)

    def _job_pipeline_add_fields(self):
        """
        Pipeline stage to add start_time, end_time, and duration fields,
        as well as translate promoted flag into promoted status.
        """
        return [
            {
                "$addFields": {
                    # Use record creation time if no start_time is
                    # available so pending jobs show up first
                    "start_time": {
                        "$cond": {
                            "if": {
                                "$and": [
                                    {
                                        "$eq": [
                                            "$status",
                                            DatabaseStatusEnum.queued,
                                        ]
                                    },
                                    {
                                        "$eq": [
                                            {"$ifNull": ["$metadata.start_time", None]},
                                            None,
                                        ]
                                    },
                                ]
                            },
                            "then": "$created_at",
                            "else": "$metadata.start_time",
                        }
                    },
                    # Compile end_time from completion_time and cancellation_time
                    "end_time": {
                        "$cond": {
                            "if": {"$eq": ["$status", "completed"]},
                            "then": "$metadata.completion_time",
                            "else": {
                                "$cond": {
                                    "if": {"$eq": ["$status", "canceled"]},
                                    "then": "$metadata.cancellation_time",
                                    "else": None,
                                }
                            },
                        }
                    },
                    # Translate promoted flag to promoted status
                    "status_merged": {
                        "$cond": {
                            "if": {"$eq": ["$promoted", PromotionStatus.COMPLETED]},
                            "then": "deployed",
                            # "else": "$status",
                            # Not sure if it makes sense to distill the temporary values...
                            "else": {
                                "$cond": {
                                    "if": {
                                        "$eq": [
                                            "$promoted",
                                            PromotionStatus.IN_PROGRESS,
                                        ]
                                    },
                                    "then": "deploying",
                                    "else": {
                                        "$cond": {
                                            "if": {
                                                "$eq": [
                                                    "$promoted",
                                                    PromotionStatus.DELETING,
                                                ]
                                            },
                                            "then": "retracting",
                                            "else": {
                                                "$cond": {
                                                    "if": {
                                                        "$and": [
                                                            {
                                                                "$eq": [
                                                                    "$status",
                                                                    DatabaseStatusEnum.canceled,
                                                                ]
                                                            },
                                                            {
                                                                # Workaround, as this doesn't work:
                                                                # { "$ne": [ "$metadata.training_duration", None ] }
                                                                "$ne": [
                                                                    {
                                                                        "$ifNull": [
                                                                            "$metadata.training_duration",
                                                                            "__MISSING__",
                                                                        ]
                                                                    },
                                                                    "__MISSING__",
                                                                ]
                                                            },
                                                        ]
                                                    },
                                                    "then": "ended",
                                                    "else": "$status",
                                                }
                                            },
                                        }
                                    },
                                }
                            },
                        }
                    },
                }
            },
            {
                "$addFields": {
                    # Compile duration from start_time and end_time
                    "duration": {
                        "$cond": {
                            "if": {"$and": ["$start_time", "$end_time"]},
                            "then": {"$subtract": ["$end_time", "$start_time"]},
                            "else": {
                                "$cond": {
                                    "if": {
                                        "$and": [
                                            "$start_time",
                                            {"$eq": ["$status", "running"]},
                                        ]
                                    },
                                    "then": {
                                        "$subtract": [
                                            datetime.now(tz=timezone.utc),
                                            "$start_time",
                                        ]
                                    },
                                    "else": None,
                                }
                            },
                        }
                    },
                }
            },
        ]

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from the database."""
        job_doc = await self.get_job(job_id)
        if job_doc:
            logger.debug(f"Archiving Job document for: {job_id}")
            await self.archived_jobs_collection.insert_one(job_doc.model_dump())
        result = await self.jobs_collection.delete_one({"job_id": job_id})
        return result.deleted_count > 0

    async def delete_metrics(self, job_id: str) -> bool:
        """Delete a metrics from the database."""
        logger.debug(f"deleting metrics: {job_id}")
        result = await self.metrics_collection.delete_one({"job_id": job_id})
        return result.deleted_count > 0

    async def insert_dataset(
        self,
        user_id: str,
        job_id: str,
        dataset: DatasetTypes,
        dataset_name: str,
        description: str,
    ) -> DatasetModel:
        """Insert a dataset from the database."""
        db_doc = {
            "user_id": user_id,
            "dataset": dataset.model_dump(),
            "dataset_name": dataset_name,
            "job_ref": [job_id],
            "description": description,
            "created_at": datetime.now(tz=timezone.utc),
        }
        await self.datasets_collection.insert_one(db_doc)
        # ! db_doc gets updated inplace with _id
        return DatasetModel(**db_doc)

    async def get_user_datasets_all(self, user_id: str) -> list[DatasetModel]:
        """Retrieve all datasets for a specific user."""
        datasets = await self.datasets_collection.find({"user_id": user_id}).to_list(
            length=None
        )
        if datasets:
            return [
                DatasetModel(**dataset) for dataset in datasets
            ]  # Pydantic serialization
        return []

    async def get_user_dataset(
        self, user_id: str, dataset_id: str
    ) -> DatasetModel | None:
        """Retrieve a single dataset for a specific user."""
        try:
            dataset = await self.datasets_collection.find_one(
                {"_id": ObjectId(dataset_id), "user_id": user_id}
            )
            if dataset:
                return DatasetModel(**dataset)  # Pydantic serialization
        except Exception as e:
            logger.error(str(e))
        return None

    async def get_user_datasets_page(
        self,
        user_id: str,
        page: int,
        page_size: int,
        sort: str,
        query: str,
        limit: list[int] | None,
    ) -> DatasetPage:
        """
        Retrieve paginated, sorted datasets for a specific user.

        Prepare the data to be consumed by the frontend table component.
        """

        # Formulate query
        # ----------------

        skip = (page - 1) * page_size
        mongo_query = {"user_id": user_id}

        # Filter by search query
        if query:
            mongo_query["dataset_name"] = {"$regex": query, "$options": "i"}

        # Formulate Sort
        # ----------------

        # Default sort key
        sort = sort or "-start_time"

        # Sort order
        sort_order = 1
        if sort and sort.startswith("-"):
            sort_order = -1
            sort = sort[1:]

        # Query pipeline
        # ----------------

        pipeline = [{"$match": mongo_query}]
        pipeline = pipeline + [
            # Add index field, needed for the table
            {
                "$setWindowFields": {
                    "sortBy": {sort: sort_order},
                    "output": {"index_": {"$documentNumber": {}}},
                }
            },
            # Make index start from zero
            # {"$addFields": {"index_": {"$subtract": ["$index", 1]}}},
            # Sort by the requested field
            {"$sort": {sort: sort_order}},
            # Parse the job_ref field to get the job names from the job table
            {
                "$lookup": {
                    "from": "jobs",
                    "localField": "job_ref",
                    "foreignField": "job_id",
                    "as": "job_ref_details",
                }
            },
            {
                "$addFields": {
                    "job_ref_names": {
                        "$map": {
                            "input": "$job_ref_details",
                            "as": "job",
                            "in": "$$job.job_name",
                        }
                    }
                }
            },
            {"$unset": "job_ref_details"},
        ]

        # This lets user limit the results to only the selected items
        if limit is not None:
            pipeline.append({"$match": {"index_": {"$in": limit}}})

        # Pagination
        pipeline = pipeline + [
            {"$skip": skip},
            {"$limit": page_size},
        ]

        # Fetch
        cursor = self.datasets_collection.aggregate(pipeline)

        # Assemble output
        # ----------------

        total_items = await self.datasets_collection.count_documents(mongo_query)
        total_items = (total_items + page_size - 1) // page_size

        items = []
        async for item in cursor:
            items.append(DatasetModel(**item))

        return DatasetPage(items=items, total=total_items, total_pages=total_items)

    async def update_dataset(
        self, user_id: str, dataset_id: str, job_id: str
    ) -> DatasetModel | None:
        """Insert a dataset from the database."""
        try:
            db_doc = {"user_id": user_id, "_id": ObjectId(dataset_id)}
            db_info = await self.datasets_collection.find_one(db_doc)
            if db_info:
                db_model = DatasetModel(**db_info)
                # update as a set only
                if job_id not in db_model.job_ref:
                    db_model.job_ref.append(job_id)
                    result = await self.datasets_collection.update_one(
                        db_doc, {"$set": {"job_ref": db_model.job_ref}}
                    )
                return db_model
        except Exception as e:
            logger.error(str(e))
        return None

    async def delete_dataset(self, user_id, dataset: str) -> bool:
        """Delete a dataset from the database."""
        result = await self.datasets_collection.delete_one(
            {"_id": ObjectId(dataset), "user_id": user_id}
        )
        return result.deleted_count > 0


# Global MongoDB manager instance
db_manager = MongoDBManager()
