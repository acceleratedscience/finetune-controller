import logging

from kubernetes import client

from app.core.config import settings
from app.core.device_config import device_configuration
from app.schemas.jobs_schemas import JobInput

# Create a logger
logger = logging.getLogger(__name__)


class PyTorchJobDeployer:
    def __init__(self, namespace: str = "default"):
        """Initialize the deployer with kubernetes configuration."""
        self.namespace = namespace
        self.k8s_custom_api = client.CustomObjectsApi()
        self.k8s_core_api = client.CoreV1Api()

    def create_pytorch_job(
        self,
        job: JobInput,
    ) -> dict:
        """Create PyTorchJob with S3 backup sidecar."""
        # model vars
        model = job.model
        selected_worker = job.device
        model_command = model.run_cmd()
        # create done.txt to signal sync container to quit
        model_command[-1] = (
            f"{model_command[-1]} && touch {model.checkpoint_mount}/done.txt"
        )
        image_pull_secret = (
            [{"name": model.image_pull_secret}] if model.image_pull_secret else []
        )
        # get defined resource worker node info from `config.json`
        worker_node = device_configuration.get_worker(selected_worker)
        # get device tolerations
        tolerations: dict = worker_node.get_tolerations()
        # get default device training resources
        resource_limits = worker_node.defaults.get_resources()
        # update default resources with new ones
        resource_limits.update(model.resources)
        # add gpu resources if any
        accelerators = worker_node.defaults.get_accelerators()
        if accelerators:
            for accelerator_key in accelerators.keys():
                for resource_key in resource_limits.keys():
                    resource_limits[resource_key].update(
                        {accelerator_key: model.accelerator_count}
                    )
        elif model.accelerator_count:
            # no specific accelerator defined in config.yaml, setting default since device requested.
            resource_limits["requests"]["nvidia.com/gpu"] = model.accelerator_count
            resource_limits["limits"]["nvidia.com/gpu"] = model.accelerator_count
        # additional pod labels / metadata
        labels = {
            # job data
            "job.owner": str(job.user_id),
            "job.db-collection": str(settings.MONGODB_DATABASE),
            "job.model-name": str(job.model_name),
            "job.instance-type": str(selected_worker),
            "job.accelerators": str(model.accelerator_count * model.cluster_nodes),
            "job.nodes": str(model.cluster_nodes),
        }
        # Kueue Scheduling. Add to LocalQueue if defined
        if worker_node.local_queue:
            labels.update({"kueue.x-k8s.io/queue-name": worker_node.local_queue})
        # dataset volume image
        init_containers = (
            [
                {
                    "name": "dataset-downloader",
                    "image": "amazon/aws-cli:latest",
                    "command": ["/bin/sh", "-c"],
                    "args": [
                        f"aws s3 cp {job.s3_uri} /data/dataset/ ; echo 'done'; find /data/dataset -type f | wc -l; ls -la /data/dataset"
                    ],
                    "volumeMounts": [
                        {"name": "aws-credentials", "mountPath": "/root/.aws"},
                        {"name": "dataset-volume", "mountPath": model.dataset_mount},
                    ],
                    "envFrom": [{"secretRef": {"name": settings.AWS_SECRET_NAME}}],
                    "env": [
                        {"name": "AWS_DEFAULT_REGION", "value": settings.AWS_REGION}
                    ],
                }
            ]
            if job.s3_uri
            else []
        )  # only use container if dataset exists
        # pytorch container for training
        main_container = {
            "name": "pytorch",
            "image": model.image,
            "command": model_command,
            "volumeMounts": [
                {
                    "name": "model-checkpoint-volume",
                    "mountPath": model.checkpoint_mount,
                    "readOnly": False,
                },
                {
                    "name": "dataset-volume",
                    "mountPath": model.dataset_mount,
                },
                {
                    "name": "dshm",
                    "mountPath": "/dev/shm",
                },
            ],
            "resources": resource_limits,
            "env": [
                {"name": "HYDRA_FULL_ERROR", "value": "1"},
                {"name": "NCCL_DEBUG", "value": "INFO"},
                {"name": "LOGLEVEL", "value": "DEBUG"},
                {"name": "PL_VERBOSE_LOGGING", "value": "1"},
            ],
        }

        # file patterns to include in sync
        include_patterns = " ".join(
            f"--include '{pattern}'" for pattern in model.store_asset_patterns
        )

        # volume to sync metrics and artifacts
        if (
            model.checkpoint_mount
            and settings.AWS_SECRET_NAME
            and settings.AWS_REGION
            and settings.S3_BUCKET_NAME
        ):
            # aws sync command
            aws_sync_cmd = f"aws s3 sync {model.checkpoint_mount} {job.s3_artifacts_uri} --exclude '*' {include_patterns} --exclude 'done.txt'"
            # build aws container sync command loop
            sync_args = [
                "-c",
                f"while [ ! -f {model.checkpoint_mount}/done.txt ]; do {aws_sync_cmd} && sleep {settings.AWS_JOB_SYNC_INTERVAL}; done; {aws_sync_cmd}; ls -la; echo 'Training complete. Exiting sidecar.';",
            ]
            aws_sync_container = {
                "name": "s3-sync",
                "image": "amazon/aws-cli:latest",
                "command": ["/bin/sh"],
                "args": sync_args,
                "volumeMounts": [
                    {
                        "name": "model-checkpoint-volume",
                        "mountPath": model.checkpoint_mount,
                        "readOnly": True,
                    },
                    {
                        "name": "aws-credentials",
                        "mountPath": "/root/.aws",
                    },
                ],
                "envFrom": [{"secretRef": {"name": settings.AWS_SECRET_NAME}}],
                "env": [
                    {
                        "name": "AWS_DEFAULT_REGION",
                        "value": settings.AWS_REGION,
                    }
                ],
            }
        else:
            logger.error(
                f"{model.checkpoint_mount=}  {settings.AWS_SECRET_NAME=}  {settings.AWS_REGION=}  {settings.S3_BUCKET_NAME=}"
            )
            raise ValueError("aws not setup corrently")

        job_manifest = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": job.job_id,
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": {
                "runPolicy": {
                    "suspend": True
                    if worker_node.local_queue
                    else False,  # Kueue https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjobs/#pytorchjob-definition
                    "backoffLimit": 2,
                    "cleanPodPolicy": "None",
                },  # cleanPodPolicy: None - keeps failed pods
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "metadata": {"labels": labels},
                            "spec": {
                                "imagePullSecrets": image_pull_secret,
                                "volumes": [
                                    {"name": "dataset-volume", "emptyDir": {}},
                                    {"name": "model-checkpoint-volume", "emptyDir": {}},
                                    {
                                        "name": "aws-credentials",
                                        "secret": {
                                            "secretName": settings.AWS_SECRET_NAME
                                        },
                                    },
                                    # !important: sizeLimit may neen to be set or resource contentions may occur
                                    {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                                ],
                                "initContainers": init_containers,
                                "containers": [main_container, aws_sync_container],
                                "imagePullPolicy": "Always",
                                "tolerations": tolerations,
                            },
                        },
                    },
                    **(
                        {
                            "Worker": {
                                "replicas": model.cluster_nodes - 1,
                                "template": {
                                    "metadata": {"labels": labels},
                                    "spec": {
                                        "imagePullSecrets": image_pull_secret,
                                        "volumes": [
                                            {"name": "dataset-volume", "emptyDir": {}},
                                            {
                                                "name": "model-checkpoint-volume",
                                                "emptyDir": {},
                                            },
                                            {
                                                "name": "aws-credentials",
                                                "secret": {
                                                    "secretName": settings.AWS_SECRET_NAME
                                                },
                                            },
                                            # !important: sizeLimit may neen to be set or resource contentiosn may occur
                                            {
                                                "name": "dshm",
                                                "emptyDir": {"medium": "Memory"},
                                            },
                                        ],
                                        "initContainers": init_containers,
                                        "containers": [main_container],
                                        "imagePullPolicy": "Always",
                                        "tolerations": tolerations,
                                    },
                                },
                            }
                        }
                        if model.cluster_nodes > 1
                        else {}
                    ),
                },
            },
        }
        # KubeflowOrgV1PyTorchJobSpec()
        # KubeflowOrgV1ElasticPolicy()
        # submit the finetune job to kubeflow training operator
        return self.k8s_custom_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=self.namespace,
            plural="pytorchjobs",
            body=job_manifest,
        )

    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a PyTorchJob."""
        return self.k8s_custom_api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=self.namespace,
            plural="pytorchjobs",
            name=job_id,
        )

    def delete_job(self, job_id: str) -> dict:
        """Delete a PyTorchJob."""
        return self.k8s_custom_api.delete_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=self.namespace,
            plural="pytorchjobs",
            name=job_id,
        )
