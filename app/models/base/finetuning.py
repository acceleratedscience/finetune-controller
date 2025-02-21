from typing import get_origin, get_args
from abc import abstractmethod, ABC
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class TrainingTask(Enum):
    REGRESSION: str = "regression"
    CLASSIFICATION: str = "classification"
    MULTITASK_CLASSIFICATION: str = "multitask_classification"


class TrainingFramework(Enum):
    PYTORCH: str = "pytorch"
    TENSORFLOW: str = "tensorflow"


class TrainingArguments(BaseModel):
    """Configuration for training"""

    model_config = ConfigDict(
        # If you want to allow extra fields
        extra="ignore"  # or 'allow', 'ignore', 'forbid'
    )


class TrainingResources(BaseModel):
    model_config = ConfigDict(
        # If you want to allow extra fields
        extra="forbid"  # or 'allow', 'ignore', 'forbid'
    )
    requests: dict[str, str | int]
    limits: dict[str, str | int] = {}  # Optional


class TrainingDataset(BaseModel):
    model_config = ConfigDict(
        # If you want to allow extra fields
        extra="forbid"  # or 'allow', 'ignore', 'forbid'
    )
    description: str = Field(default="")
    dataset_required: bool = Field(
        default=False, description="Whether a dataset is required for training"
    )
    dataset_name: str = Field(
        default="", description="Name of dataset file from api. set at runtime."
    )


class BaseFineTuneModel(BaseModel, ABC):
    model_config = ConfigDict(
        # If you want to allow extra fields
        extra="ignore"  # or 'ignore', 'forbid'
    )
    # model setup
    name: str = Field(..., min_length=4, pattern=r"^[a-zA-Z0-9._@]+$")
    image: str
    image_pull_secret: str | None = None
    command: list[str]
    framework: TrainingFramework
    task: TrainingTask
    description: str = ""
    project_url: str = ""

    # defaults
    checkpoint_mount: str = Field(
        default="/data/artifacts",
        description="Mount point for storing results. Best not to change this.",
    )
    dataset_mount: str = Field(
        default="/data/dataset",
        description="Mount point for storing dataset. Best not to change this.",
    )
    dataset_info: TrainingDataset = TrainingDataset()
    device_types: list[str] = Field(
        default=["cpu"],
        description="Node type to run on based on taint toleration. Default 'cpu' (normal) worker node",
    )
    resources: TrainingResources = TrainingResources(
        requests={"cpu": 2, "memory": "1Gi"}
    )
    accelerator_count: int = Field(
        default=1,
        ge=1,
        description="Number of gpu devices to use for training per worker",
    )
    cluster_nodes: int = Field(
        default=1, ge=1, description="Total number of workers for training"
    )
    store_asset_patterns: list[str] = Field(
        default=["*.json", "*.yaml", "*.csv", "*.pt", "*.ckpt"],
        description="Pattern match a list of files to store.",
    )
    promotion_path: str = Field(
        default="",
        description="s3 prefix to upload artifacts. Based on Inference path `domain/algorithm_name/algorithm_application`",
    )

    # Training configuration
    training_arguments: TrainingArguments

    @abstractmethod
    def run_cmd(self) -> list[str]:
        pass

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Ensures that overridden default values in child classes maintain the same type.
        """
        super().__init_subclass__(**kwargs)

        parent_fields = BaseFineTuneModel.model_fields
        for field_name, field_info in parent_fields.items():
            base_type = field_info.annotation
            child_type = cls.__annotations__.get(
                field_name, base_type
            )  # Get declared type

            base_origin = get_origin(base_type)
            child_origin = get_origin(child_type)

            # Handle generic types
            if base_origin or child_origin:
                base_args = get_args(base_type)
                child_args = get_args(child_type)

                if base_origin != child_origin or set(base_args) != set(child_args):
                    raise TypeError(
                        f"Field '{field_name}' in {cls.__name__} must have type {base_type}, but got {child_type}"
                    )
            else:
                # Regular type check
                if isinstance(base_type, type) and isinstance(child_type, type):
                    if (
                        not issubclass(child_type, base_type)
                        and child_type != base_type
                    ):
                        raise TypeError(
                            f"Field '{field_name}' in {cls.__name__} must have type {base_type}, but got {child_type}"
                        )


if __name__ == "__main__":

    class StableDiffusionFineTuneModel(BaseFineTuneModel):
        name: str = ""
        image: str = "image"
        command: list[str] = []
        framework: TrainingFramework = TrainingFramework.PYTORCH
        task: TrainingTask = TrainingTask.CLASSIFICATION
        test1: str = ""

        def run_cmd(self):
            # build your command logic
            return f"accelerate launch train_dreambooth.py --pretrained_model_name_or_path={self.image} --instance_data_dir=/data/instance --output_dir=/data/output --instance_prompt='a photo of {self.name}' --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=1e-6 --lr_scheduler='constant' --lr_warmup_steps=0 --max_train_steps=1000".split(
                " "
            )

    model = StableDiffusionFineTuneModel(command=[])
    print(model.run_cmd())
