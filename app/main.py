import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal
import traceback
import asyncio

from fastapi import (
    APIRouter,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse, StreamingResponse
from kubernetes import client
from kubernetes.client.rest import ApiException
from pydantic import ValidationError, HttpUrl
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.device_config import DeviceTypes
from app.database.db import db_manager
from app.schemas.db_schemas import (
    JobsPage,
    DatabaseStatusEnum,
    PromotionStatus,
    DatasetPage,
    JobStatus,
)
from app.schemas.kubeflow_schemas import TrainingJobStatus
from app.schemas.jobs_schemas import (
    Job,
    JobInput,
    PaginatedTableResponse,
    JobIdsRequest,
    JobMetaData,
    DatasetInput,
    DatasetMeta,
    Dataset,
)
from app.core.monitor import job_monitor
from app.jobs.registered_models import JOB_MANIFESTS, load_model_modules
from app.jobs.task_builder import task_builder
from app.models.base.finetuning import BaseFineTuneModel, TrainingTask
from app.utils.kube_config import api_instance
from app.utils.kube_helpers import get_pod_events, get_pod_status
from app.utils.logging_config import setup_logging
from app.utils.naming import generate_short_uuid
from app.utils.S3Handler import s3_handler
from app.api.middleware import setup_middleware, limiter
from app.api.custom_openapi import custom_openapi_jwt_auth
from app.core.security import UserJWT, JWTExtraInfo, dev_generate_token, verify_token
from app.utils.stream_logger import LogStreamManager
from app.tasks.promotion import PromotionTask

# region --- Setup

# Set up logging configuration
setup_logging()


# Create a logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown tasks"""
    logger.info(f"Running in {settings.ENVIRONMENT} environment")
    if settings.ENVIRONMENT == "local":
        # Log all settings in local environment
        for key, value in settings.model_dump().items():
            logger.debug(f" ENV > {key}: {value}")
    # Startup: connect to the database
    await db_manager.connect()
    # load modules
    load_model_modules()
    # enable monitor for local development in namespace, api + monitor, same instance deployment
    if settings.DEV_LOCAL_JOB_MONITOR:
        logger.warning(
            "Local Job Monitor Enabled. Beware wont scale unless deployed seperately"
        )
        await job_monitor.start()
    else:
        logger.info(
            f"Local Job Monitor Disabled. Ensure Job Monitor Deployed On Cluster Namspace: {settings.NAMESPACE}"
        )
    yield
    # Shutdown: close the connection
    await db_manager.close()
    if settings.DEV_LOCAL_JOB_MONITOR or job_monitor.monitoring_task:
        await job_monitor.stop()


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Add Middleware
setup_middleware(app)

# add custom openapi schema
app.openapi = custom_openapi_jwt_auth(app)


# Create a prefixed v1 API
api_v1 = APIRouter(prefix=settings.API_V1_STR)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPExceptions globally"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    logger.debug(f"Rate Limit IP: {get_remote_address(request)}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "status_code": exc.status_code},
    )


# endregion
# region --- JWT Validation


def decode_request(
    request: Request,
) -> tuple[JWTExtraInfo, UserJWT] | tuple[None, None]:
    # See OpenBridgeBasicMiddleware
    jwt_data = getattr(request.state, "jwt_data", None)
    jwt_decoded = getattr(request.state, "decoded_jwt", None)
    return jwt_data, jwt_decoded


def validate_user_access(jwt: UserJWT, db_record: JobStatus) -> None:
    """Raises error if user not owner of resource"""
    if jwt and db_record and jwt.user_id != db_record.user_id:
        logger.warning(
            f"user ({jwt.user_id}) tried to access restricted resource ({db_record.job_id})"
        )
        raise HTTPException(status_code=400, detail="Cannot access resource")


# endregion
# region --- Development


if settings.ENVIRONMENT != "production":
    # Bridge-user cookie for development
    # See OpenBridgeBasicMiddleware for validation
    @app.get("/auth/cookie", tags=["Auth"])
    @limiter.limit("10/minute")
    async def get_dev_bridge_user_cookie(
        request: Request, response: Response, user=Query("default_user")
    ):
        """Return a spoofed bridge-user cookie for development"""

        # Generate JWT token
        token = await dev_generate_token(user)

        # Wrap it in dict to mimmick the bridge-user cookie
        bridge_user_cookie = json.dumps(
            {
                "config": None,
                "resources": [],
                "subject": "foobar",
                "user_type": "group",
                "token": token,
            }
        )

        # Attach cookie
        response.set_cookie(
            key="bridge-user",
            value=bridge_user_cookie,
            httponly="True",
            samesite="Strict",
        )
        return token

    @app.get("/auth/generate", tags=["Auth"])
    @limiter.limit("10/minute")
    async def generate_token_auth(
        request: Request,
        response: Response,
        user: str = Query("default_user"),
        include_models: str = Query(""),
    ):
        """Generate JWT

        Args:
            user (str, optional): user to authenticate. Defaults to "default_user".
            include_models (str, optional): specific models to include. Defaults to all available models.
        """
        models = [m.strip() for m in include_models.split(",") if m]
        logger.warning(models)
        token = await dev_generate_token(user, models)
        return {"token": token}

    @app.get("/auth/verify", tags=["Auth"])
    @limiter.limit("10/minute")
    async def verify_token_auth(request: Request, response: Response, token: str):
        """Get authorization"""
        payload = await verify_token(token)
        return payload


# endregion
# region --- General


@app.get("/health")
@api_v1.get("/health")
@limiter.limit("20/minute")
async def health(
    request: Request,
):
    """Health Check"""
    logger.debug("health check")
    return {"status": "ok"}


@api_v1.get("/models", tags=["Models"])
@limiter.limit("20/minute")
async def list_available_models(
    request: Request,
) -> dict[str, Any]:
    """Return the available options for the create-job form"""
    # validate access
    jwt_data, jwt = decode_request(request)
    if jwt_data and jwt:
        # get all available models
        models = user_available_models(jwt)
    else:
        models = user_available_models()

    # create model form data
    form_data = {}
    for model_name in models:
        try:
            # create instance of class
            model_cls: BaseFineTuneModel = await get_model(model_name)
            # get training arguments
            arguments = model_cls.training_arguments.model_json_schema()
            form_data[model_name] = {
                "description": model_cls.description,
                "url": model_cls.project_url,
                "arguments": arguments.get("properties", {}),
                "task": model_cls.task,
                # "devices": model_cls.device_types,
                "devices": list(DeviceTypes),  # list all devices
                "dataset_required": model_cls.dataset_info.dataset_required,
                "dataset_description": model_cls.dataset_info.description,
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Could not get form data for model ({model_name}) {str(e)}")
            continue
    return form_data


@api_v1.get("/models/{model}", tags=["Models"])
@limiter.limit("50/minute")
async def get_model_details(
    model: str,
    request: Request,
):
    """Get the optional parameters and their default values for a specific model"""
    # validate access
    jwt_data, jwt = decode_request(request)
    try:
        if jwt_data and model not in user_available_models(jwt):
            logger.warning(
                f"user ({jwt.user_id}) tried to access restricted resource ({model})"
            )
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        # Get the model class from JOB_MANIFESTS
        model_class = JOB_MANIFESTS.get(model)
        if not model_class:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

        # Create an instance to get the default values
        model_instance: BaseFineTuneModel = model_class()

        # Get the model's schema which includes all fields
        schema = model_instance.model_json_schema()

        # Filter for optional fields and their defaults
        optional_params = {}
        model_detail = {}
        for field_name, field in schema["properties"].items():
            # Check if field has a default value and is not a required field
            if field_name not in schema.get("required", []) and field_name not in [
                "name",
                "image",
                "image_pull_secret",
                "command",
                "framework",
                "description",
                "checkpoint_mount",
                "dataset_mount",
                "task",
                "dataset_name",
            ]:
                optional_params[field_name] = getattr(model_instance, field_name)
            elif field_name in ["framework", "description", "task"]:
                model_detail[field_name] = getattr(model_instance, field_name)

        return {"arguments": optional_params} | model_detail
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model options: {str(e)}"
        ) from e


@api_v1.websocket("/logs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str, full_log=True, follow=True):
    """WebSocket endpoint handler"""
    logger.debug(f"Connecting WebSocket for job {job_id}")
    await websocket.accept()

    stream_manager = LogStreamManager(websocket, job_id, full_log, follow)
    try:
        await stream_manager.run()
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"Unexpected error in stream_job: {str(e)}")


# endregion
# region --- Jobs


DEFAULT_USER = "default_user"


# Start job
@api_v1.post("/jobs", tags=["Jobs"])
@limiter.limit("10/minute")
async def start_job(
    request: Request,
    user_id: str = Form(
        DEFAULT_USER,
        pattern=r"^[a-zA-Z0-9._@]+$",
        min_length=4,
        description="Used when BearerAuth not provided. when left blank: default_user",
    ),
    # fmt: off
    job_name: str = Form(..., description="Name for job"),
    model: str = Form(..., description="Model to finetune"),  # type: ignore
    device: DeviceTypes = Form(..., description="Name of device to train on"),  # type: ignore
    task: TrainingTask = Form(..., description="Name of task type"),
    arguments: str = Form("{}", description="Training arguments"),
    dataset_description: str | None = Form(
        "", description="Description for this dataset."
    ),
    dataset_id: str | Literal[""] | None = Form(
        "", description="Use an existing uploaded dataset"
    ),
    dataset_url: HttpUrl | Literal[""] | None = Form(
        "", description="Upload dataset from url"
    ),
    dataset: UploadFile | Literal[""] | None = File(
        None, description="Upload dataset from file"
    ),
    # fmt: on
):
    """Start a finetune job"""
    # validate access
    jwt_data, jwt = decode_request(request)
    if jwt_data and jwt:
        # replace user_id with jwt provided user id
        user_id = jwt.user_id
        if model not in user_available_models(jwt):
            logger.warning(
                f"user ({jwt.user_id}) tried to access restricted resource ({model})"
            )
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    # convert json to python dict
    model_arguments = _parse_arguments_input(arguments)

    model_name = model
    job_id = f"{model_name.strip().lower().replace('_', '-')}-{generate_short_uuid()}"  # Generate unique job ID

    # check dataset input type
    if dataset_id:
        dataset_input = DatasetInput(dataset_id=dataset_id)
    elif dataset_url:
        dataset_input = DatasetInput(dataset_url=dataset_url)
    elif dataset:
        dataset_input = DatasetInput(dataset_file=dataset)
    else:
        # no dataset requested
        dataset_input = DatasetInput()
    # update description from input
    dataset_input.dataset_description = dataset_description

    try:
        # get an instance of pydantic model
        model_class = JOB_MANIFESTS.get(model)
        if not model_class:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

        # create a model instance with updated arguments
        model_instance = model_class.model_validate(
            model_class(training_arguments=model_arguments)
        )
        # convert user input to correct object types (pydantic validation)
        model_arguments = model_instance.training_arguments.model_dump()

        # verify device
        # if device.name not in model_instance.device_types:
        #     raise HTTPException(status_code=400, detail="Device not available")

        # validate the task corresponds to the right model.
        if task != model_instance.task:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task ({task.name}) for model ({model_instance.task.name})",
            )
    except ValidationError as e:
        # Format validation errors into a readable message
        error_messages = []
        for error in e.errors():
            # show last item in iterable as error key
            field = error["loc"][-1] if error["loc"] else "unknown field"
            message = error["msg"]
            error_messages.append(f"{field}: {message}")
        raise HTTPException(
            status_code=400,
            detail=f"<b class='regular'>Invalid model parameters:</b><br>• {'<br>• '.join(error_messages)}",
        ) from e

    # Create meta job object
    job_input = JobInput(
        user_id=user_id,
        job_name=job_name,
        model_name=model_name,
        model=model_instance,
        device=device.name,
        arguments=model_arguments if model_arguments else None,
        job_id=job_id,
    )

    try:
        # Submit the PyTorchJob to the Kubeflow operator
        training_job = await task_builder(job_input, settings.NAMESPACE, dataset_input)
        logger.info(f"Job started successfully: {job_input.job_id}")
        return {"message": "Job started successfully", "job_id": job_input.job_id}
    except ApiException as e:
        logger.error(str(e), exc_info=True)
        dataset_info = _printable_dataset_info(dataset_input)
        raise HTTPException(
            status_code=e.status,
            detail=f"Failed to start job {job_name} / {job_id}<br>{e}{dataset_info}",
        ) from e
    except Exception as e:
        logger.error(str(e), exc_info=True)
        dataset_info = _printable_dataset_info(dataset_input)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start job {job_name} / {job_id}<br>{e}<br>{dataset_info}",
        ) from e


def _parse_arguments_input(arguments: str = Form(...)) -> dict[str, Any]:
    """Parse JSON string to dictionary"""
    try:
        args = json.loads(arguments)
        return args
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON for arguments") from e


def _printable_dataset_info(dataset_input):
    """Format dataset information for error messages"""
    dataset_info = {
        k: v for k, v in dict(dataset_input).items() if v and k != "dataset_description"
    }
    dataset_info = "<br>".join([f"{k}: {v}" for k, v in dataset_info.items()])
    return dataset_info


# Paginated jobs for table
@api_v1.get("/jobs", tags=["Jobs"])
@limiter.limit("50/minute")
async def get_user_jobs_page(
    request: Request,
    user_id: str = Query(DEFAULT_USER),  # For debugging
    page: int = Query(1),  # Page number
    page_size: int = Query(10),  # Page size
    sort: str = Query(None),  # Sort key
    query: str = Query(None),  # Search string
    limit: str = Query(None),  # Limit results to list of indices, eg. ?limit=1,7,12
    status: str = Query(None),  # Filter by status
    model_name: str = Query(None),  # Filter by model_name
):
    """Get one page of job statuses for a user"""
    # validate access
    jwt_data, jwt = decode_request(request)
    # Get user_id from JWT if provided
    if jwt_data and jwt:
        user_id = jwt.user_id
    try:
        logger.info(f"Getting jobs for user {user_id}")

        # Parse limit query parameter into list
        # ?limit=1,2,3 --> [1,2,3]
        limit = [int(x) for x in limit.split(",")] if limit else None

        # Get job information from database
        jobs_data: JobsPage = await db_manager.get_user_jobs(
            user_id, page, page_size, sort, query, limit, status, model_name
        )

        # Compile list of jobs as return data
        items = []
        for job in jobs_data.items:
            model_cls: BaseFineTuneModel = await get_model(job.model_name)
            promotion_path = (
                model_cls.promotion_path.strip("/")
                if model_cls and model_cls.promotion_path
                else "Not available"
            )
            items.append(
                Job(
                    index_=job.index_,
                    job_id=job.job_id,
                    job_name=job.job_name,
                    promoted=job.promoted,
                    model_name=job.model_name,
                    queue_pos=job.metadata.queue_pos,
                    status=job.status,
                    status_merged=job.status_merged,
                    start_time=job.start_time,
                    end_time=job.end_time,
                    duration=job.duration,
                    dataset_id=job.dataset_id,
                    #
                    meta_={
                        "error": None,
                        "note": None,
                        "data": JobMetaData(
                            job_name=job.job_name,
                            job_id=job.job_id,
                            model_name=job.model_name,
                            promotion_path=promotion_path,
                            device=job.device,
                            task=job.task,
                            framework=job.framework,
                            arguments=job.arguments or "-",
                            dataset_name=job.dataset_name or "-",
                        ),
                    },
                )
            )

        # For API testing
        # return [f"{item.index_}--{item.start_time}--{item.name}--{item.id}" for item in items]

        return PaginatedTableResponse(
            total=jobs_data.total,
            totalPages=jobs_data.total_pages,
            resultIndices=[],
            page=page,
            pageSize=page_size,
            items=items,
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        ) from e


# Get single job
@api_v1.get("/jobs/{job_id}", tags=["Jobs"])
@limiter.limit("50/minute")
async def get_job(
    request: Request,
    job_id: str,
):
    """Get the status of a finetune job"""
    # validate access
    jwt_data, jwt = decode_request(request)
    try:
        job_info = await db_manager.get_job(job_id)
        validate_user_access(jwt, job_info)

        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        logger.info("Getting job status for %s", job_id)
        return job_info

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        ) from e


async def get_key_url(data: list[dict], key="metrics.csv"):
    """
    Retrieve the URL for a given key from a list of dictionaries.

    :param data: List of dictionaries containing 'key' and 'url'.
    :param key: The key to search for.
    :return: The corresponding URL if found, otherwise None.
    """
    for item in data:
        if item.get("key") == key:
            return item.get("url")
    return None


# Metrics
@api_v1.get("/jobs/{job_id}/metrics", tags=["Jobs"])
@limiter.limit("50/minute")
async def get_job_metrics(
    request: Request,
    job_id: str,
):
    """Get metrics from the metrics.json file for a specific job"""
    logger.info(f"requesting metrics for job id ({job_id})")
    # validate access
    jwt_data, jwt = decode_request(request)
    try:
        # Get job status
        job_info = await db_manager.get_job(job_id)
        validate_user_access(jwt, job_info)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # get job metrics
        job_metrics = await db_manager.get_job_metrics(job_id)

        # check state
        if not job_metrics and job_info.status in TrainingJobStatus.running_states:
            raise HTTPException(
                status_code=202, detail="Job is still running. no metrics found"
            )
        if not job_metrics and job_info.status in TrainingJobStatus.stopped_states:
            raise HTTPException(
                status_code=404,
                detail=f"no metrics available for job {job_id}",
            )
        else:
            # Show latest epochs first and limit to
            # 100 items not to overload frontend
            job_metrics.metrics.reverse()
            job_metrics.metrics = job_metrics.metrics[:100]

            metrics_data = job_metrics.model_dump()
            try:
                # Get presigned URLs for artifacts
                urls = await s3_handler.get_presigned_urls(job_metrics.user_id, job_id)
                metrics_data["metrics_url"] = await get_key_url(urls, "metrics.csv")
            except Exception as e:
                logger.error(f"Could not get presigned urls for job {job_id}: {str(e)}")
                metrics_data["metrics_url"] = None
            return metrics_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Action -- promote
@api_v1.post("/jobs/{job_id}/promote", tags=["Jobs"])
@limiter.limit("2/minute")
async def promote_job(
    request: Request,
    job_id: str,
    background_tasks: BackgroundTasks,
):
    """Promote a job"""
    # validate access
    jwt_data, jwt = decode_request(request)
    try:
        # Get job status
        job_info = await db_manager.get_job(job_id)
        validate_user_access(jwt, job_info)
        # TODO: implement buckets for each user group
        # if jwt_data:
        #     # get bucket from users group id
        #     bucket_name = jwt_data.group_id
        # else:
        #     # test ducket deployment
        #     bucket_name = settings.S3_DEFAULT_DEPLOY_BUCKET.strip()

        # use default s3 bucket
        bucket_name = settings.S3_DEFAULT_DEPLOY_BUCKET.strip()

        # validate job
        if not job_info:
            await asyncio.sleep(0.5)
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        if job_info.promoted == PromotionStatus.IN_PROGRESS:
            await asyncio.sleep(0.5)
            raise HTTPException(status_code=202, detail="Job is being promoted already")
        if job_info.status in TrainingJobStatus.running_states:
            await asyncio.sleep(0.5)
            raise HTTPException(status_code=200, detail="Cannot promote running job")
        if not job_info.atrifacts_uri or not bucket_name:
            await asyncio.sleep(0.5)
            raise HTTPException(status_code=404, detail="Cannot promote this job")

        # get instance of model class
        model_cls: BaseFineTuneModel = await get_model(job_info.model_name)

        # check if model already promoted
        if job_info.promoted == PromotionStatus.COMPLETED:
            return {
                "status": PromotionStatus.COMPLETED,
                "model": model_cls.promotion_path.strip("/"),
                "version": job_id,
            }

        # validate model has promotion prefix path
        if not model_cls.promotion_path:
            raise HTTPException(status_code=400, detail="Model cannot be promoted")

        # model version uses name as job_id
        # create promotion uri (where to move artifacts to)
        destination_uri = "s3://" + "/".join(
            [bucket_name, model_cls.promotion_path.strip("/"), job_id]
        )

        logger.info(f"promoting job ({job_id})")

        # Add task to background tasks
        background_tasks.add_task(
            PromotionTask.promote_job_task,
            job_id,
            job_info.atrifacts_uri,
            destination_uri,
        )

        return {
            "status": "promotion_initiated",
            "job_id": job_id,
            "message": "Job promotion started in background",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Action -- unpromote
@api_v1.post("/jobs/{job_id}/unpromote", tags=["Jobs"])
@limiter.limit("2/minute")
async def unpromote_job(
    request: Request,
    background_tasks: BackgroundTasks,
    job_id: str,
):
    jwt_data, jwt = decode_request(request)
    # check db status
    job_info = await db_manager.get_job(job_id)

    # validate access
    validate_user_access(jwt, job_info)

    if not job_info:
        raise HTTPException(status_code=400, detail="Job not found")
    if not job_info.promoted == PromotionStatus.COMPLETED:
        raise HTTPException(
            status_code=400, detail="Model not promoted. cannot unpromote"
        )
    if not job_info.destination_uri:
        logger.warning(f"model {job_id} promoted but cannot unpromote")
        raise HTTPException(status_code=400, detail="Cannot unpromote model")

    # unpromote job
    logger.info(f"unpromoting job {job_id}")
    # Add task to background tasks
    background_tasks.add_task(
        PromotionTask.unpromote_job_task,
        job_id,
        job_info.destination_uri,
    )

    return {
        "status": "unpromotion_initiated",
        "job_id": job_id,
        "message": "Job unpromotion started in background",
    }


# Action -- cancel
@api_v1.post("/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(request: Request, job_id: str):
    """Cancel a job"""
    jwt_data, jwt = decode_request(request)
    try:
        job_info = await db_manager.get_job(job_id)
        validate_user_access(jwt, job_info)

        if job_info.status in TrainingJobStatus.stopped_states:
            raise HTTPException(status_code=409, detail="Item already cancelled")
        # Delete the PyTorchJob
        try:
            api_instance.delete_namespaced_custom_object(
                group="kubeflow.org",
                version="v1",
                namespace=settings.NAMESPACE,
                plural="pytorchjobs",
                name=job_id,
                body=client.V1DeleteOptions(),
            )
        except ApiException as e:
            if e.status == 404:
                # Job has already been completed
                logger.debug(f"Job {job_id} not found. could not delete from cluster.")
            else:
                logger.error("Failed to delete PyTorchJob %s: %s", job_id, e)
                raise e

        # Update job status in MongoDB
        now = datetime.now(timezone.utc)  # UTC
        await db_manager.update_job_status(
            job_id=job_id,
            status=DatabaseStatusEnum.canceled.value,
            metadata={
                "cancellation_time": now,
                "message": "Job canceled by user",
            },
        )

        # Fetch updated job info to update frontend
        job_info = await db_manager.get_job(job_id)
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        logger.info("Job canceled successfully: %s", job_id)

        # Return job information
        return {
            "status": job_info.status,
            "start_time": job_info.start_time,
            "end_time": job_info.end_time,
            "duration": job_info.duration,
        }

    except HTTPException as e:
        raise e
    except ApiException as e:
        logger.error("Failed to cancel job %s: %s", job_id, e)
        raise HTTPException(
            status_code=e.status, detail=f"Failed to cancel job {job_id}: {e}"
        ) from e

    except Exception as e:
        logger.error("Unexpected error while canceling job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}") from e


# Action -- delete
@api_v1.delete("/jobs/delete", tags=["Jobs"])
async def delete_job(request: Request, body: JobIdsRequest):
    """Delete one    ore more jobs from the database"""

    # Validate access
    jwt_data, jwt = decode_request(request)

    # Endpoint return object
    output = {}

    for job_id in body.job_ids:
        logger.debug(f"Deleting job {job_id}")
        job_info = await db_manager.get_job(job_id)

        validate_user_access(jwt, job_info)

        # Not found
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # Still running
        if job_info.status in TrainingJobStatus.running_states:
            raise HTTPException(status_code=400, detail="Job is still running")

        # Delete operations
        if job_info.promoted == PromotionStatus.COMPLETED and job_info.destination_uri:
            # delete promoted bucket files
            await s3_handler.cleanup_uri_items(job_info.destination_uri)
        if job_info.atrifacts_uri:
            # delete finetune bucket artifacts
            await s3_handler.cleanup_uri_items(job_info.atrifacts_uri)
        # delete metrics if any
        await db_manager.delete_metrics(job_id)
        # remove from database
        await db_manager.delete_job(job_id)

        # Compile return
        output["job_id"] = {"message": "Jobs deleted successfully"}

    return output


# endregion
# region --- Datasets


@api_v1.get("/datasets/all", tags=["Datasets"])
@limiter.limit("30/minute")
async def get_user_datasets_all(request: Request, user_id: str = Query(DEFAULT_USER)):
    """Get all datasets for a user, to populate a dropdown"""
    # validate user
    jwt_data, jwt = decode_request(request)
    if jwt and jwt_data:
        user_id = jwt.user_id

    datasets = await db_manager.get_user_datasets_all(user_id)
    # return the dataset detail but not the s3 path
    return [dataset.model_dump(exclude={"dataset": {"s3_uri"}}) for dataset in datasets]


@api_v1.get("/datasets", tags=["Datasets"])
@limiter.limit("30/minute")
async def get_user_datasets_page(
    request: Request,
    user_id: str = Query(DEFAULT_USER),
    page: int = Query(1),  # Page number
    page_size: int = Query(10),  # Page size
    sort: str = Query(None),  # Sort key
    query: str = Query(None),  # Search string
    limit: str = Query(None),  # Limit results to list of indices, eg. ?limit=1,7,12
):
    """Get one page of datasets for a user"""

    # Get user_id from JWT if provided
    jwt_data, jwt = decode_request(request)
    if jwt:
        user_id = jwt.user_id

    try:
        # Parse limit query parameter into list
        # ?limit=1,2,3 --> [1,2,3]
        limit = [int(x) for x in limit.split(",")] if limit else None

        # Gete datasets information from databse
        datasets_data: DatasetPage = await db_manager.get_user_datasets_page(
            user_id, page, page_size, sort, query, limit
        )

        # Compile list of jobs as return data
        items = []
        for item in datasets_data.items:
            # Assemble metadata
            _meta_data = {
                "Id": item.id,
                "Related jobs": (
                    ", ".join(item.job_ref_names)
                    if len(item.job_ref_names) > 0
                    else "-"
                ),
            }
            if item.dataset.http_url:
                _meta_data["Source"] = item.dataset.http_url
            meta_ = DatasetMeta(
                error=None,
                note=item.description,
                data=_meta_data,
            )

            items.append(
                Dataset(
                    meta_=meta_,
                    **item.model_dump(
                        include={
                            "index_",
                            "id",
                            "dataset_name",
                            "description",
                            "created_at",
                        },
                    ),
                )
            )

        return PaginatedTableResponse(
            total=datasets_data.total,
            totalPages=datasets_data.total_pages,
            resultIndices=[],
            page=page,
            pageSize=page_size,
            items=items,
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get datasets: {str(e)}"
        ) from e


@api_v1.delete("/datasets/{dataset_id}", tags=["Datasets"])
async def delete_datasets(
    request: Request, dataset_id: str, user_id: str = Query(DEFAULT_USER)
):
    logger.debug(f"deleting dataset {dataset_id} for user {user_id}")
    jwt_data, jwt = decode_request(request)
    if jwt:
        user_id = jwt.user_id

    job_info = await db_manager.delete_dataset(user_id, dataset_id)
    return job_info


# @api_v1.put("/dataset", tags=["Datasets"])
# async def update_datasets(
#     request: Request,
#     job_id: str,
#     dataset_id: str,
#     user_id: str = Query(DEFAULT_USER),
# ):
#     jwt_data, jwt = decode_request(request)
#     if jwt:
#         user_id = jwt.user_id
#     dataset = await db_manager.update_dataset(user_id, dataset_id, job_id)
#     return dataset


# @api_v1.post("/dataset", tags=["Datasets"])
# async def insert_datasets(
#     request: Request,
#     job_id: str,
#     dataset: DatasetTypes,
#     dataset_name: str,
#     description: str,
#     user_id: str = Query(DEFAULT_USER),
# ):
#     jwt_data, jwt = decode_request(request)
#     if jwt:
#         user_id = jwt.user_id
#     dataset = await db_manager.insert_dataset(
#         user_id, job_id, dataset, dataset_name, description
#     )
#     return dataset


# endregion
# region --- Admin


@api_v1.get("/admin/artifacts/{job_id}", tags=["Admin"])
@limiter.limit("5/minute")
async def get_artifacts(
    request: Request,
    job_id: str,
    user_id: str = Query(DEFAULT_USER),
):
    """Download artifacts for a specific job as a zip file"""
    jwt_data, jwt = decode_request(request)
    if jwt:
        # replace query with jwt user_id
        user_id = jwt.user_id
    logger.info(
        f"User ({user_id}), requesting artifacts download for job id ({job_id})"
    )
    temp_dir = None
    try:
        # Get job information from database
        job_info = await db_manager.get_job(job_id)
        job_status = job_info.status

        # Check if job has completed
        if job_status in TrainingJobStatus.running_states:
            logger.debug(job_status)
            raise HTTPException(
                status_code=400,
                detail="Job is still running or has not completed successfully",
            )

        # Download and zip artifacts
        zip_path, temp_dir = await s3_handler.download_artifacts(user_id, job_id)

        def cleanup_and_stream():
            try:
                with open(zip_path, "rb") as f:
                    yield from f
            finally:
                # Cleanup temporary directory and all its contents
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        return StreamingResponse(
            cleanup_and_stream(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=artifacts_{job_id}.zip"
            },
        )
    except HTTPException as e:
        raise e
    except ApiException as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if e.status == 404:
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found"
            ) from e
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        ) from e
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail=str(e)) from e


@api_v1.get("/admin/artifacts/presigned_urls/{job_id}", tags=["Admin"])
@limiter.limit("10/minute")
async def get_artifact_urls(
    request: Request,
    job_id: str,
    user_id: str = Query(DEFAULT_USER),
):
    """Get presigned URLs for all artifacts in a job"""
    jwt_data, jwt = decode_request(request)
    if jwt:
        # replace user id with jwt user id
        user_id = jwt.user_id
    logger.info(f"User ({user_id}), requesting artifact URLs for job id ({job_id})")
    try:
        # Get job information from database
        job_info = await db_manager.get_job(job_id)
        # Not found
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        job_status = job_info.status

        # Check if job has completed
        if job_status in TrainingJobStatus.running_states:
            logger.debug(job_status)
            raise HTTPException(
                status_code=400,
                detail="Job is still running or has not completed successfully",
            )

        # Get presigned URLs for artifacts
        urls = await s3_handler.get_presigned_urls(user_id, job_id)
        return {"artifacts": urls}
    except HTTPException as e:
        raise e
    except ApiException as e:
        logger.error(str(e))
        if e.status == 404:
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found"
            ) from e
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e


@api_v1.get("/admin/job/poll/{job_id}", tags=["Admin"])
async def poll_admin_job(job_id: str):
    """More detailed info about a job"""
    try:
        logger.info(f"Getting job status for {job_id}")
        # Get the PyTorchJob status
        api_response = api_instance.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=settings.NAMESPACE,
            plural="pytorchjobs",
            name=job_id,
        )
        selector = api_response["status"]["replicaStatuses"]["Master"]["selector"]
        logger.debug(f"Selector: {selector}")
        events = await get_pod_events(selector, settings.NAMESPACE)
        pod_status = await get_pod_status(selector, settings.NAMESPACE)
        job_status: dict = api_response["status"]["conditions"][-1]
        job_status["events"] = [
            {"type": event.type, "message": event.message, "reason": event.reason}
            for event in events
        ]
        if pod_status:
            job_status.update(pod_status)
        return {"status": job_status}
    except HTTPException as e:
        raise e
    except ApiException as e:
        logger.error(str(e))
        if e.status == 404:
            raise HTTPException(
                status_code=404, detail=f"Job '{job_id}' not found"
            ) from e
        raise HTTPException(
            status_code=500, detail=f"Failed to get job status: '{job_id}'"
        ) from e
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e


@api_v1.get("/admin/jobs/list", tags=["Admin"])
async def list_jobs():
    """List jobs on cluster"""
    try:
        api_response = api_instance.list_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=settings.NAMESPACE,
            plural="pytorchjobs",
        )
        jobs = [job["metadata"]["name"] for job in api_response["items"]]
        logger.info(f"Jobs list: {jobs}")
        return {"jobs": jobs}
    except HTTPException as e:
        raise e
    except ApiException as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}") from e


@api_v1.delete("/admin/jobs/{user_id}", tags=["Admin"])
async def clean_user_jobs(user_id):
    """Remove all user jobs from database"""
    try:
        logger.debug(f"Cleaning all user {user_id} Jobs")
        jobs = await db_manager.get_all_user_jobs(user_id)
        if jobs:
            jobs_list = []
            for job in jobs:
                jobs_list.append(job.job_id)
                await delete_job(job.job_id)
            return {
                "message": "Jobs deleted successfully",
                "jobs": jobs_list,
            }
        else:
            return {
                "message": "No jobs deleted",
                "jobs": [],
            }
    except HTTPException as e:
        raise e
    except ApiException as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete jobs: {e}")


# For development only, can be deleted later
# TODO: remove
@api_v1.get("/sample-data.csv")
async def csv_sample():
    """Return sample data CSV file"""
    from fastapi.responses import Response

    csv_data = """Id,SMILES,esol
aaa,CCC(=O)C=C(N)C(C)(C)c1nccs1,2.1
bbb,CCOC(=O)CC(=O)C(C)(C)c1ccccn1,9.2
ccc,CCOC(=O)CC(N)C(C)(C)c1ccccn1,6.5"""
    return Response(content=csv_data, media_type="text/csv")


# Include the API router in the main app
app.include_router(api_v1)


# ------------------------
# endregion
# region --- Methods


def user_available_models(token: UserJWT | None = None):
    """Get all available models"""
    all_models: list = list(JOB_MANIFESTS.keys())
    # Filter models based on user available models.
    # If the user has specific available models, only include those.
    # get all models if running local for testing
    if token:
        user_models = [
            user_model
            for user_model in token.available_models
            if user_model in all_models
        ]
        return user_models
    # Access all models if jwt not set. Authorization handled by Middleware
    return all_models


async def get_model(model_name: str) -> BaseFineTuneModel:
    """Create instance of model class"""
    model_class = JOB_MANIFESTS.get(model_name)
    return model_class() if model_class else None  # type: ignore


# endregion

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
