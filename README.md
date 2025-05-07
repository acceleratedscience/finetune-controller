# Finetune Controller

Finetune Controller is a robust and flexible system designed to manage and streamline the fine-tuning of machine learning models on Kubernetes, particularly within OpenShift clusters. This project leverages modern tools and workflows, enabling efficient development and deployment processes for AI-driven applications.

### Features
- Local Development: Get started quickly with a streamlined setup process using uv, a high-performance Python package and project manager.
- OpenShift Integration: Simplify deployment and scaling with OpenShift-specific configurations and GPU support for intensive workloads.
- MongoDB Backend: Seamlessly connect to a local or cluster-based MongoDB database.
- Extensibility: Easily integrate with the Kubeflow Training Operator and other components for advanced workflows.

## Getting Started

If the cluster is already set up continue else follow the cluster setup instructions [here](#setup-openshift-cluster)
### Prereqs
1. Recommend using [uv](https://github.com/astral-sh/uv), *an extremely fast Python package and project manager*

    ```shell
    pip install uv
    ```
2. A container engine such as Docker or Podman

<!-- ## Quick Setup using Compose

1. Start the application using Docker Compose
    ```shell
    docker compose up -d
    ```

This will:
- Start MongoDB with the required configuration
- Build and start the FastAPI server
- Make the application available at http://localhost:8000

To view logs of the controller:
```shell
docker compose logs controller -f
```

To stop the application:
```shell
docker compose down
``` -->


### Install
1. Create virtual environment and install dependencies
    ```shell
    uv sync
    ```

2. Start a local developement mongo database *(or connect to one on cluster with port-forward)*

    Local
    ```shell
    docker run -d --rm --name mongodb \
        -e MONGODB_INITDB_ROOT_USERNAME="default-user" \
        -e MONGODB_INITDB_ROOT_PASSWORD="admin123456789" \
        -e MONGODB_INITDB_DATABASE="finetune" \
        -p 27017:27017 \
        mongodb/mongodb-community-server:latest
    ```

    you can port-forward this connection to your local machine
    ```shell
    oc port-forward service/mongodb-community-server 27017:27017 -n <namespace>
    ```

3. Connect to the Openshift cluster with the cli login command `oc login`. If cluster not already set up follow [these](#setup-openshift-cluster) steps

4. Create a project level `.env` file (see `.env.example`) and update the variables.
    ```shell
    cp .env.example .env
    ```

5. Make sure the virtual environment is activated and start the local finetuning controller application.
    ```shell
    source .venv/bin/activate

    uvicorn app.main:app --reload
    ```

This will:
- Start MongoDB with the required configuration
- Build and start the FastAPI server
- Make the application available at http://localhost:8000

### Development and Contributing
Setup [pre-commit](https://pre-commit.com/#install) to keep linting and code styling up to standard.
```shell
uv sync
pre-commit install
```

## Setup OpenShift Cluster Resources

### Create default project
Name can be descriptive for these examples we will use `finetune-controller`
```shell
oc new-project finetune-controller
```

### Create Kubeflow project
```shell
oc new-project kubeflow
```

### Install Kubeflow training operator
<!-- ```shell
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
``` -->
```shell
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

### Install Kueue
> Requires Kubernetes 1.29 or newer

Follow the latest [docs](https://kueue.sigs.k8s.io/docs/installation/)

Install a released version
```shell
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml
```

To wait for Kueue to be fully available, run:
```shell
kubectl wait deploy/kueue-controller-manager -nkueue-system --for=condition=available --timeout=5m
```

Restart pods
```shell
kubectl delete pods -lcontrol-plane=controller-manager -nkueue-system
```

First update the namepspace for the crd LocalQueue object in [default-user-queue.yaml](/crds/kueue/default-user-queue.yaml). default namepsace: "default"
```shell
yq e '.metadata.namespace = "finetune-controller"' -i crds/kueue/default-user-queue.yaml
```

Apply the default CRD config for Kueue or update by following their docs
```shell
kubectl apply -f crds/kueue/
```

### Install mongodb server

Example configuration. *do properly configure for production*
```shell
oc new-app -e MONGODB_INITDB_ROOT_USERNAME="default-user" -e MONGODB_INITDB_ROOT_PASSWORD="admin123456789" -e MONGODB_INITDB_DATABASE="finetune"  mongodb/mongodb-community-server:latest --namespace finetune-controller
```

<!-- ### Install Kubeflow pipelines
> reference [here](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

```shell
export PIPELINE_VERSION=2.3.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

update the pipeline ui to disable gke metadata due to bug [here](https://github.com/kubeflow/pipelines/issues/11247)
```shell
oc set env deployment/ml-pipeline-ui DISABLE_GKE_METADATA=true
``` -->

### Add GPU nodes to ROSA cluster
Go to your cluster on redhat console [admin dashboard](https://console.redhat.com/openshift/cluster-list). Add a machine pool of your choosing with the following configuration:

Taints
```
key: nvidia.com/gpu
value: <machine pool type or other>
effect: NoSchedule
```

Node Labels
```
Key: cluster-api/accelerator
Value: <gpu type e.g. V100 or empty>
```

### Setup AWS Secret

Example aws config
```yaml
# aws_credentials.yaml
apiVersion: v1
data:
  AWS_ACCESS_KEY_ID: |base64 encoded secret
  AWS_SECRET_ACCESS_KEY: |base64 encoded secret
  AWS_REGION: |base64 encoded string
kind: Secret
metadata:
  name: aws-credentials
type: Opaque

```

Example for base 64 command in terminal
```bash
echo -n "VALUE" | base64
```

### Setup Pull secrets

Example docker pull secret config
```yaml
# pull_secret.yaml
apiVersion: v1
data:
  .dockerconfigjson: ...
kind: Secret
metadata:
  name: cr-pull-secret
type: kubernetes.io/dockerconfigjson

```

Apply these secrets
```shell
oc apply -f aws-credentials.yaml -n finetune-controller
```

## Install Finetune Controller On OpenShift

1. Create a `.env.production` file and update the defaults. For this example set `MONGODB_URL=mongodb://mongodb-community-server.finetune-controller.svc.cluster.local:27017`
    ```shell
    cp .env.example .env.production
    ```

2. create the application
    ```shell
    oc new-app --strategy=docker --binary --name finetune-controller --env-file=".env.production" --namespace finetune-controller
    ```

3. expose services and patch tls config
    ```shell
    oc expose deployment/finetune-controller --port=8000
    oc expose svc/finetune-controller --port=8000
    oc patch route finetune-controller --type=merge -p '{"spec":{"tls":{"termination":"edge"}}}'
    ```

4. add cluster role binding permissions to the application

5. start a build
    ```shell
    oc start-build finetune-controller --from-dir=. --namespace=finetune-controller
    ```

## Manually Publish Updates To Finetune Controller
Publish From current project
```shell
./scripts/publish.sh
```
Publish From git ~HEAD
```shell
./scripts/publish_git.sh
```
