# Setting Up Finetuning Models

This guide explains how to add and configure custom finetuning models in the finetune-controller project.

## Table of Contents

1. [Model Architecture Overview](#model-architecture-overview)
2. [Creating a Custom Model](#creating-a-custom-model)
3. [Model Registration](#model-registration)
4. [Directory Structure](#directory-structure)
5. [Configuration Options](#configuration-options)
6. [Best Practices](#best-practices)
7. [Example Walkthrough](#example-walkthrough)

## Model Architecture Overview

The finetune-controller uses a modular architecture where each finetuning model is defined as a Python class that inherits from `BaseFineTuneModel`. The system automatically discovers and registers models from the `app/models/custom/` directory.

### Key Components

- **BaseFineTuneModel**: Base class that all finetuning models must inherit from
- **TrainingArguments**: Configuration class for model-specific training parameters
- **Model Registration**: Automatic discovery and registration system
- **Dynamic Loading**: Runtime loading of custom models

## Creating a Custom Model

To create a custom finetuning model, follow these steps:

### Step 1: Create Training Configuration Class

First, define a configuration class that inherits from `TrainingArguments`:

```python
from app.models.base.finetuning import TrainingArguments, Field

class MyModelConfig(TrainingArguments):
    """Configuration parameters for your custom model"""

    # Define model-specific parameters with defaults
    batch_size: int = Field(
        default=32,
        description="Size of each batch during training"
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate for the optimizer"
    )
    epochs: int = Field(
        default=10,
        description="Number of training epochs"
    )
    # Add more parameters as needed
```

### Step 2: Create Model Class

Define your model class inheriting from `BaseFineTuneModel`:

```python
from app.models.base.finetuning import (
    BaseFineTuneModel,
    TrainingFramework,
    TrainingTask,
    Field,
    TrainingResources,
    TrainingDataset,
)

class MyCustomModel(BaseFineTuneModel):
    """Custom finetuning model specification"""

    # Required fields
    name: str = "MyCustomModel"  # Display name in frontend
    inference_name: str | None = "MyCustomModel"  # Must match inference service name
    description: str = "Description of your custom model"
    project_url: str = "https://github.com/your-org/your-model"
    image: str = "your-registry/your-model:latest"  # Container image
    command: list[str] = [
        "/bin/bash",
        "-c",
        "python train.py",  # Your training script
    ]

    # Model metadata
    framework: TrainingFramework = TrainingFramework.PYTORCH  # or TENSORFLOW
    task: TrainingTask = TrainingTask.CLASSIFICATION  # or other task types

    # Dataset configuration
    dataset_info: TrainingDataset = TrainingDataset(
        description="Description of expected dataset format",
        dataset_required=True  # Set to False if no dataset needed
    )

    # Resource requirements
    resources: TrainingResources = TrainingResources(
        requests={"cpu": 2, "memory": "4Gi"},
        limits={"cpu": 4, "memory": "8Gi"}
    )

    # GPU configuration
    accelerator_count: int = Field(
        default=1,
        ge=0,
        description="Number of GPU devices per worker"
    )

    # Model promotion path (S3 storage path)
    promotion_path: str = Field(
        default="domain/algorithm_name/application",
        description="S3 path format: domain/algorithm_name/algorithm_application"
    )

    # Training configuration
    training_arguments: MyModelConfig = MyModelConfig()

    def run_cmd(self) -> list[str]:
        """Convert model properties to command arguments"""
        cmd = self.command.copy()
        args = []

        # Add training arguments
        args.append(f"--batch-size={self.training_arguments.batch_size}")
        args.append(f"--learning-rate={self.training_arguments.learning_rate}")
        args.append(f"--epochs={self.training_arguments.epochs}")

        # Required mount paths (always include these)
        args.append(f"--dataset_path={self.dataset_mount}")
        args.append(f"--checkpoint_path={self.checkpoint_mount}")

        # Combine command with arguments
        cmd[-1] += " " + " ".join(args)
        return cmd
```

### Step 3: Place in Custom Directory

Save your model file in the `app/models/custom/` directory:

```
app/models/custom/my_custom_model.py
```

## Model Registration

The system automatically discovers and registers models through the following process:

### Automatic Discovery

1. **Loading Process**: The `load_model_modules()` function in `app/jobs/registered_models.py` scans the `app/models/custom/` directory
2. **Dynamic Import**: Uses `load_models_from_directory()` from `app/models/model_loader.py` to import model classes
3. **Registration**: Adds discovered models to the `JOB_MANIFESTS` dictionary using the model's `name` field as the key

### Registration Code Flow

```python
# In registered_models.py
def load_model_modules():
    # Load custom models from directory
    custom_models_dir = Path(__file__).parent.parent / "models" / "custom"
    custom_models = load_models_from_directory(str(custom_models_dir))

    # Register each model
    for model_name, model_class in custom_models.items():
        name = model_class.model_fields.get("name").get_default()
        if name not in JOB_MANIFESTS:
            JOB_MANIFESTS[name] = model_class
```

### Manual Registration (Alternative)

You can also manually register models by adding them to the `JOB_MANIFESTS` dictionary:

```python
from app.models.custom.my_custom_model import MyCustomModel

JOB_MANIFESTS = {
    "MyCustomModel": MyCustomModel,
    # ... other models
}
```

## Directory Structure

```
app/
├── models/
│   ├── base/
│   │   └── finetuning.py          # Base classes
│   ├── examples/
│   │   └── mnist.py               # Example model
│   └── custom/                    # Your custom models go here
│       ├── __init__.py
│       ├── .gitkeep
│       └── your_model.py          # Your custom model files
├── jobs/
│   └── registered_models.py       # Model registration
└── ...
```

## Configuration Options

### Required Fields

| Field       | Type                | Description                                                                 |
| ----------- | ------------------- | --------------------------------------------------------------------------- |
| `name`      | `str`               | Display name in frontend (must be unique, min 4 chars, alphanumeric + .\_@) |
| `image`     | `str`               | Container image with your training code                                     |
| `command`   | `list[str]`         | Command to execute training                                                 |
| `framework` | `TrainingFramework` | ML framework (PYTORCH, TENSORFLOW)                                          |
| `task`      | `TrainingTask`      | Task type (CLASSIFICATION, REGRESSION, MULTITASK_CLASSIFICATION)            |

### Optional Fields

| Field                  | Type                | Default                                           | Description                                                                          |
| ---------------------- | ------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `description`          | `str`               | `""`                                              | Model description                                                                    |
| `project_url`          | `str`               | `""`                                              | Project repository URL                                                               |
| `inference_name`       | `str \| None`       | `None`                                            | Name for inference service integration. Allows for many to one inference service     |
| `image_pull_secret`    | `str \| None`       | `None`                                            | Secret name for pulling private container images                                     |
| `checkpoint_mount`     | `str`               | `"/data/artifacts"`                               | Mount point for storing results. Best not to change this.                            |
| `dataset_mount`        | `str`               | `"/data/dataset"`                                 | Mount point for storing dataset. Best not to change this.                            |
| `dataset_info`         | `TrainingDataset`   | `TrainingDataset()`                               | Dataset configuration (see TrainingDataset fields below)                             |
| `device_types`         | `list[str]`         | `["cpu"]`                                         | Node type to run on based on taint toleration. Default 'cpu' (normal) worker node    |
| `resources`            | `TrainingResources` | `{"requests": {"cpu": 2, "memory": "1Gi"}}`       | CPU/Memory requirements                                                              |
| `accelerator_count`    | `int`               | `1`                                               | Number of GPU devices per worker (minimum 1)                                         |
| `cluster_nodes`        | `int`               | `1`                                               | Total number of workers for training (minimum 1)                                     |
| `store_asset_patterns` | `list[str]`         | `["*.json", "*.yaml", "*.csv", "*.pt", "*.ckpt"]` | Pattern match a list of files to store                                               |
| `promotion_path`       | `str`               | `""`                                              | S3 prefix to upload artifacts. Format: `domain/algorithm_name/algorithm_application` |

### TrainingDataset Configuration

The `TrainingDataset` class defines dataset requirements and metadata:

| Field              | Type   | Default | Description                                    |
| ------------------ | ------ | ------- | ---------------------------------------------- |
| `description`      | `str`  | `""`    | Description of the expected dataset format     |
| `dataset_required` | `bool` | `False` | Whether a dataset is required for training     |
| `dataset_name`     | `str`  | `""`    | Name of dataset file from API (set at runtime) |

Example:

```python
dataset_info: TrainingDataset = TrainingDataset(
    description="Expects CSV files with columns: image_path, label",
    dataset_required=True
)
```

### TrainingResources Configuration

The `TrainingResources` class defines CPU and memory requirements:

| Field      | Type                    | Required | Description                        |
| ---------- | ----------------------- | -------- | ---------------------------------- |
| `requests` | `dict[str, str \| int]` | Yes      | Minimum resource requirements      |
| `limits`   | `dict[str, str \| int]` | No       | Maximum resource limits (optional) |

Example:

```python
resources: TrainingResources = TrainingResources(
    requests={"cpu": 4, "memory": "2Gi"},
    limits={"cpu": 8, "memory": "4Gi"}
)
```

Common resource specifications:

- **CPU**: Integer or string (e.g., `2`, `"2"`, `"2000m"` for 2 cores)
- **Memory**: String with unit (e.g., `"1Gi"`, `"2048Mi"`, `"2G"`)

### Training Arguments

Define custom training parameters by creating a class that inherits from `TrainingArguments`:

```python
class CustomConfig(TrainingArguments):
    # Each parameter must have a default value and type annotation
    param_name: type = Field(
        default=default_value,
        description="Parameter description",
        # Optional Field constraints:
        ge=0,  # Greater than or equal
        le=100,  # Less than or equal
        # ... other pydantic Field options
    )
```

## Best Practices

### 1. Model Naming

- Use descriptive, unique names
- Follow PascalCase for class names
- Use consistent naming between `name` and `inference_name`

### 2. Container Images

- Include all dependencies in your container image
- Use specific version tags, avoid `:latest` in production
- Test your container image independently before integration

### 3. Resource Management

- Set appropriate resource requests and limits
- Consider GPU requirements realistically
- Test resource usage with representative workloads

### 4. Parameter Configuration

- Provide sensible defaults for all parameters
- Include helpful descriptions
- Use appropriate Field constraints

### 5. Command Generation

- Always include dataset and checkpoint mount paths
- Handle boolean flags appropriately
- Ensure argument parsing matches your training script

### 6. Error Handling

- Test model loading with the `__main__` block pattern
- Validate configurations before deployment
- Include logging for debugging

## Example Walkthrough

Let's examine the MNIST example to understand the complete implementation:

### MNIST Configuration Class

```python
class MNISTConfig(TrainingArguments):
    """Model Params for MNIST Finetune Job"""

    batch_size: int = Field(default=64, description="Size of each batch during training")
    test_batch_size: int = Field(default=1000, description="Size of each batch during testing")
    epochs: int = Field(default=1, description="Number of epochs for training")
    lr: float = Field(default=1.0, description="Learning rate for the optimizer")
    gamma: float = Field(default=0.7, description="Learning rate step gamma for scheduler")
    no_cuda: bool = Field(default=False, description="Disable CUDA (use CPU instead of GPU)")
    seed: int = Field(default=1, description="Random seed for reproducibility")
    log_interval: int = Field(default=10, description="How many batches to wait before logging training status")
    save_model: bool = Field(default=False, description="Whether to save the trained model")
```

### MNIST Model Class

```python
class MNIST(BaseFineTuneModel):
    """Finetune Job Spec for MNIST"""

    name: str = "MNIST"
    inference_name: str | None = "MNIST"
    description: str = "Example MNIST model for fine-tuning"
    project_url: str = "https://github.com/acceleratedscience/model-foobar"
    image: str = "quay.io/brian_duenas/mnist:latest"
    command: list[str] = ["/bin/bash", "-c", "python mnist_training_script.py"]

    framework: TrainingFramework = TrainingFramework.PYTORCH
    task: TrainingTask = TrainingTask.CLASSIFICATION

    dataset_info: TrainingDataset = TrainingDataset(
        description="MNIST model does not expect a dataset",
        dataset_required=False
    )

    resources: TrainingResources = TrainingResources(
        requests={"cpu": 4, "memory": "1Gi"},
        limits={"cpu": 8, "memory": "2Gi"}
    )

    accelerator_count: int = Field(default=0, ge=1, description="Number of gpu devices to use for training per worker")

    promotion_path: str = Field(
        default="molecules/mnist/mnist_test",
        description="s3 path to upload artifacts. Based on Inference path `domain/algorithm_name/algorithm_application`"
    )

    training_arguments: MNISTConfig = MNISTConfig()
```

### Command Generation

```python
def run_cmd(self) -> list[str]:
    """Converts model properties to command arguments"""
    cmd = self.command.copy()
    args = []

    # Handle boolean flags
    if self.training_arguments.no_cuda:
        args.append("--no-cuda")
    if self.training_arguments.save_model:
        args.append("--save-model")

    # Add value parameters
    args.append(f"--batch-size={self.training_arguments.batch_size}")
    args.append(f"--test-batch-size={self.training_arguments.test_batch_size}")
    args.append(f"--epochs={self.training_arguments.epochs}")
    args.append(f"--lr={self.training_arguments.lr}")
    args.append(f"--seed={self.training_arguments.seed}")
    args.append(f"--log-interval={self.training_arguments.log_interval}")
    args.append(f"--gamma={self.training_arguments.gamma}")

    # Required mount paths
    args.append(f"--dataset_path={self.dataset_mount}")
    args.append(f"--checkpoint_path={self.checkpoint_mount}")

    # Combine with base command
    cmd[-1] += " " + " ".join(args)
    return cmd
```

### Testing Your Model

Include a test block to validate your model definition:

```python
if __name__ == "__main__":
    # Test that the model definition correctly loads
    from pprint import pprint

    model = MNIST(training_arguments={"epochs": 2}, description="test model load")
    pprint(model.model_dump())
```

## Troubleshooting

### Common Issues

1. **Model Not Appearing**: Check that your model file is in `app/models/custom/` and properly inherits from `BaseFineTuneModel`

2. **Import Errors**: Ensure all required imports are available and your model class is properly defined

3. **Name Conflicts**: Each model must have a unique `name` field value

4. **Container Issues**: Verify your container image exists and contains the necessary training code

5. **Resource Problems**: Check that your resource requests are reasonable for your cluster

### Debugging

Enable debug logging to see model registration details:

```python
import logging
logging.getLogger("app.jobs.registered_models").setLevel(logging.DEBUG)
```

Run the model test block to validate your configuration:

```bash
python -m app.models.custom.your_model
```
