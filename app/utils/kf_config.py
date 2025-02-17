import logging

from kubeflow.training import TrainingClient

from app.core.config import settings

# Create a logger
logger = logging.getLogger(__name__)


# Initialize Kubeflow client api
kubeflow_api: TrainingClient = TrainingClient(namespace=settings.NAMESPACE)
