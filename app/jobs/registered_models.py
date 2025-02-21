import logging
from pathlib import Path

from app.models.base.finetuning import BaseFineTuneModel
from app.models.examples import mnist
from app.models.model_loader import load_models_from_directory

logger = logging.getLogger(__name__)


# Combine built-in models with custom models
# JOB_MANIFESTS: dict[str, BaseFineTuneModel] = {
#     mnist.MNIST.__name__: mnist.MNIST,
# }
JOB_MANIFESTS: dict[str, BaseFineTuneModel] = {
    mnist.MNIST.model_fields.get("name").get_default(): mnist.MNIST,
}


def load_model_modules():
    logger.info("Loading modules")
    # Load custom models from the custom models directory
    custom_models_dir = Path(__file__).parent.parent / "models" / "custom"
    custom_models = load_models_from_directory(str(custom_models_dir))
    # Add custom models to the manifests
    for model_name, model_class in custom_models.items():
        # Use the model's defined name if available, otherwise use the class name
        try:
            # name = getattr(model_class, "name", model_name)
            name = model_class.model_fields.get("name").get_default()
            if name in JOB_MANIFESTS:
                logger.warning(f"Model name '{name}' already registered! skipping...")
                continue
            JOB_MANIFESTS[name] = model_class
            logger.info(f"registered model: {name}")
        except Exception as e:
            logger.error(f"Failed to load model: {name}\n{str(e)}")
