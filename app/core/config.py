import logging
import base64
from typing import Literal
from pydantic import SecretStr, computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.logging_config import setup_logging
from app.utils.kube_config import core_v1_api


# configure logging for startup
setup_logging()
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="./.env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
    # server
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    API_V1_STR: str = "/api/v1"
    # cluster
    NAMESPACE: str
    # CORS
    FRONTEND_URL_CORS: list[str] = Field(default_factory=list)
    # security - Introspection URL and OpenBridge credentials
    OPENBRIDGE_INTROSPECTION_URL: str | None = None
    OPENBRIDGE_CLIENT_ID: str | None = None
    OPENBRIDGE_CLIENT_SECRET: SecretStr | None = None
    OPENBRIDGE_API_KEY: SecretStr | None = None  # Placeholder, Not Implemented.
    # Disable introspection for local development only
    DEV_DISABLE_INTROSPECTION: bool = False
    # worker config
    CONFIGURATION_FILE: str = "config.json"
    # database
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_USERNAME: SecretStr | None = None
    MONGODB_PASSWORD: SecretStr | None = None
    MONGODB_DATABASE: str = "default"
    # job monitor
    JOB_MONITOR_INTERVAL: int = 2
    DEV_LOCAL_JOB_MONITOR: bool = False
    AWS_JOB_SYNC_INTERVAL: int = 60
    # aws configuration
    S3_DEFAULT_DEPLOY_BUCKET: str = ""
    S3_BUCKET_NAME: str
    AWS_SECRET_NAME: str

    # aws credentials computed from k8s secret `AWS_SECRET_NAME`
    @computed_field(return_type=SecretStr)
    @property
    def AWS_SECRET_KEY(self) -> SecretStr:
        return SecretStr(
            base64.b64decode(
                core_v1_api.read_namespaced_secret(
                    name=settings.AWS_SECRET_NAME, namespace=settings.NAMESPACE
                ).data["AWS_SECRET_ACCESS_KEY"]
            ).decode("utf-8")
        )

    @computed_field(return_type=SecretStr)
    @property
    def AWS_ACCESS_KEY(self) -> SecretStr:
        return SecretStr(
            base64.b64decode(
                core_v1_api.read_namespaced_secret(
                    name=settings.AWS_SECRET_NAME, namespace=settings.NAMESPACE
                ).data["AWS_ACCESS_KEY_ID"]
            ).decode("utf-8")
        )

    @computed_field(return_type=str)
    @property
    def AWS_REGION(self) -> str:
        return str(
            base64.b64decode(
                core_v1_api.read_namespaced_secret(
                    name=settings.AWS_SECRET_NAME, namespace=settings.NAMESPACE
                ).data["AWS_REGION"]
            ).decode("utf-8")
        )


settings = Settings()
