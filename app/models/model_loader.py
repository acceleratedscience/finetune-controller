import importlib
import sys
from pathlib import Path

# from git import Repo
# import tempfile
from app.models.base.finetuning import BaseFineTuneModel
import logging


logger = logging.getLogger(__name__)


def load_models_from_directory(directory_path) -> dict[str, BaseFineTuneModel]:
    """
    Dynamically load models from a given directory.
    Args:
        directory_path (str): Path to the directory containing model modules.

    Returns:
        dict: Mapping of model names to loaded model classes.
    """
    models = {}
    sys.path.insert(0, directory_path)  # Temporarily add to Python path

    for file in Path(directory_path).glob("*.py"):
        if file.name.startswith("__"):  # Skip special files
            continue

        module_name = file.stem
        try:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseFineTuneModel)
                    and attr is not BaseFineTuneModel
                ):
                    models[attr.__name__] = attr
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}")

    sys.path.pop(0)  # Clean up Python path
    return models


# def clone_and_load_models(repo_url, branch="main"):
#   """Support Models from Remote Sources"""
#     with tempfile.TemporaryDirectory() as temp_dir:
#         Repo.clone_from(repo_url, temp_dir, branch=branch)
#         return load_models_from_directory(temp_dir)

# Implement Model Registry
# Maintain a registry to keep track of models and their sources for easier management. This could be as simple as a JSON file or a database table.
# {
#     "models": [
#         {
#             "name": "MNIST",
#             "source": "git",
#             "url": "https://github.com/your-org/mnist-model.git",
#             "branch": "main"
#         },
#         {
#             "name": "CustomModel",
#             "source": "local",
#             "path": "/path/to/models/custom_model.py"
#         }
#     ]
# }
