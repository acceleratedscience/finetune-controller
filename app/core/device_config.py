import logging
import json
import re
from enum import Enum
from pydantic import BaseModel
from app.core.config import settings

from app.utils.logging_config import setup_logging


# configure logging for startup
setup_logging()
logger = logging.getLogger(__name__)


class ResourceDefaults(BaseModel):
    requests: dict[str, str | int] = {"cpu": 2, "memory": "1Gi"}
    limits: dict[str, str | int] = {}  # Optional


class Toleration(BaseModel):
    key: str
    value: str
    effect: str


class Defaults(BaseModel):
    resources: ResourceDefaults = ResourceDefaults()
    accelerators: dict[str, int] = {}  # Optional

    def get_resources(self) -> dict[str, dict[str, str | int]]:
        return self.resources.model_dump()

    def get_accelerators(self):
        return self.accelerators


class Worker(BaseModel):
    name: str
    local_queue: str | None = None
    defaults: Defaults = Defaults()
    tolerations: list[Toleration] | None = None

    def get_tolerations(self):
        if self.tolerations:
            return [tol.model_dump() for tol in self.tolerations]
        return []


class WorkersConfig(BaseModel):
    workers: list[Worker] = []
    default_queue: str | None = None

    def list_workers(self) -> list[str]:
        if self.workers:
            return [worker.name for worker in self.workers]
        return []

    def get_worker(self, name) -> Worker | None:
        if self.workers:
            for worker in self.workers:
                if name == worker.name:
                    # if worker has no local queue defined assign the default queue
                    if self.default_queue and not worker.local_queue:
                        worker.local_queue = self.default_queue
                    return worker
        return None


class APIConfiguration(BaseModel):
    workers: WorkersConfig = WorkersConfig()

    def get_worker(self, name) -> Worker:
        """Main method to get a worker. Do not access a worker directly"""
        return self.workers.get_worker(name)

    def list_workers(self) -> Worker:
        return self.workers.list_workers()


def remove_json_comments(json_str):
    """JSON helper function to remove comments"""
    # Remove // comments
    json_str = re.sub(r"//.*", "", json_str)
    return json_str


def load_config() -> APIConfiguration:
    """Load config.json"""
    try:
        with open(settings.CONFIGURATION_FILE) as fp:
            config_fp = fp.read()
            config = json.loads(remove_json_comments(config_fp))
            logger.debug(f"Using configuration: {settings.CONFIGURATION_FILE}")
            return APIConfiguration(workers=WorkersConfig(**config))
    except Exception as e:
        logging.error(
            f">> Error loading ({settings.CONFIGURATION_FILE}). Create a {settings.CONFIGURATION_FILE} file to populate devices for finetuning.",
            exc_info=True,
        )
        return APIConfiguration()


device_configuration: APIConfiguration = load_config()

# enum of all available models names
DeviceTypes: list[str] = Enum(
    "DeviceTypes", {name: name for name in device_configuration.list_workers()}
)
