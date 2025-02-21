# MNIST Example from https://github.com/brian316/mnist-example
from app.models.base.finetuning import (
    BaseFineTuneModel,
    TrainingFramework,
    TrainingTask,
    Field,
    TrainingArguments,
    TrainingResources,
    TrainingDataset,
)


class MNISTConfig(TrainingArguments):
    """Model Params for MNIST Finetune Job"""

    # model params should have only 1 type and a default
    batch_size: int = Field(
        default=64, description="Size of each batch during training"
    )
    test_batch_size: int = Field(
        default=1000, description="Size of each batch during testing"
    )
    epochs: int = Field(default=1, description="Number of epochs for training")
    lr: float = Field(default=1.0, description="Learning rate for the optimizer")
    gamma: float = Field(
        default=0.7, description="Learning rate step gamma for scheduler"
    )
    no_cuda: bool = Field(
        default=False, description="Disable CUDA (use CPU instead of GPU)"
    )
    seed: int = Field(default=1, description="Random seed for reproducibility")
    log_interval: int = Field(
        default=10,
        description="How many batches to wait before logging training status",
    )
    save_model: bool = Field(
        default=False, description="Whether to save the trained model"
    )


class MNIST(BaseFineTuneModel):
    """Finetune Job Spec for MNIST"""

    name: str = "MNIST"  # model name must match inference name to work
    description: str = "Example MNIST model for fine-tuning"
    project_url: str = "https://github.com/acceleratedscience/model-foobar"
    image: str = "quay.io/brian_duenas/mnist:latest"
    command: list[str] = [
        "/bin/bash",
        "-c",
        "python mnist_training_script.py",
    ]
    framework: TrainingFramework = TrainingFramework.PYTORCH
    task: TrainingTask = TrainingTask.CLASSIFICATION
    dataset_info: TrainingDataset = TrainingDataset(
        description="MNIST model does not expect a dataset", dataset_required=False
    )
    resources: TrainingResources = TrainingResources(
        requests={"cpu": 4, "memory": "1Gi"}, limits={"cpu": 8, "memory": "2Gi"}
    )
    accelerator_count: int = Field(
        default=0,
        ge=1,
        description="Number of gpu devices to use for training per worker",
    )
    promotion_path: str = Field(
        default="molecules/mnist/mnist_test",
        description="s3 path to upload artifacts. Based on Inference path `domain/algorithm_name/algorithm_application`",
    )

    # training parameter defaults
    training_arguments: MNISTConfig = MNISTConfig()

    def run_cmd(self) -> list[str]:
        # Converts model properties to the command arguments
        cmd = self.command.copy()
        args = []

        # add training config arguments
        if self.training_arguments.no_cuda:
            args.append("--no-cuda")
        if self.training_arguments.save_model:
            args.append("--save-model")
        args.append(f"--batch-size={self.training_arguments.batch_size}")
        args.append(f"--test-batch-size={self.training_arguments.test_batch_size}")
        args.append(f"--epochs={self.training_arguments.epochs}")
        args.append(f"--lr={self.training_arguments.lr}")
        args.append(f"--seed={self.training_arguments.seed}")
        args.append(f"--log-interval={self.training_arguments.log_interval}")
        args.append(f"--gamma={self.training_arguments.gamma}")

        # !important add default dataset and checkpoint mounts
        args.append(f"--dataset_path={self.dataset_mount}")
        args.append(f"--checkpoint_path={self.checkpoint_mount}")

        # Join args to the base command string
        cmd[-1] += " " + " ".join(args)
        return cmd


if __name__ == "__main__":
    # test that the model definition correctly loads
    from pprint import pprint

    mnist = MNIST(training_arguments={"epochs": 2}, description="test model load")
    pprint(mnist.model_dump())
