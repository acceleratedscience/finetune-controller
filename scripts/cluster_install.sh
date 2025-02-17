#!/bin/bash

# Create a new project
# This script will create a new project with the default settings
# sets up cluster and namespace and deploys the default services

# defaults
NAMESPACE="default"
ENV_FILE=".env"
APP_NAME="finetune-controller"

# Create a new project
read -p "Press Enter to continue with using env file ($ENV_FILE), or type a new one: " input_env_file
if [ ! -z "$input_env_file" ]; then
    ENV_FILE=$input_env_file
fi
echo ">> Using env file: $ENV_FILE"

# Try to read NAMESPACE from .env file, default to "default" if not found
if [ -f .env ]; then
    NAMESPACE=$(grep '^NAMESPACE=' $ENV_FILE | cut -d '=' -f2 || echo "default")
    echo "Found namespace from $ENV_FILE: $NAMESPACE"
else
    NAMESPACE="default"
    echo "Could not find NAMESPACE in $ENV_FILE using namespace: $NAMESPACE"
fi

read -p "Press Enter to continue with namespace ($NAMESPACE), or type a new one: " input_namespace
if [ ! -z "$input_namespace" ]; then
    NAMESPACE=$input_namespace
fi

echo ">> Using Namespace: $NAMESPACE"
echo ">> Using App Name: $APP_NAME"

echo ">> The script will now install the following:
    - Kubflow Training operator
    - Kueue Scheduling operator
    - Mongo Database
    - Finetune-Controller App
    - add cluster-admin role to user default"


read -p "Press Enter to continue with installation... " input_continue

echo ">> Creating new project $NAMESPACE.."
oc new-project $NAMESPACE

echo ">> Creating new project kubeflow.."
oc new-project kubeflow

echo ">> Deploying Kubeflow Training Operator.."
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1" --namespace kubeflow

echo ">> Deploying Kueue Scheduler Operator"
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml
kubectl wait deploy/kueue-controller-manager -nkueue-system --for=condition=available --timeout=5m
kubectl delete pods -lcontrol-plane=controller-manager -nkueue-system

echo ">> Deploying test MongoDB instance.."
sleep 1
oc new-app -e MONGODB_INITDB_ROOT_USERNAME="default-user" -e MONGODB_INITDB_ROOT_PASSWORD="admin123456789" -e MONGODB_INITDB_DATABASE="finetune"  mongodb/mongodb-community-server:latest --namespace $NAMESPACE

# pause for user to review
read -p "Manually create the AWS_SECRET_NAME from $ENV_FILE secret to namespace $NAMESPACE in the cluster. press enter to continue.. " input_continue

echo ">> Deploying $APP_NAME application.."
oc new-app --strategy=docker --binary --name $APP_NAME --env-file=$ENV_FILE --namespace $NAMESPACE
# setup the deployment config
oc expose deployment/$APP_NAME --port=8000 --namespace $NAMESPACE
oc expose svc/$APP_NAME --port=8000 --namespace $NAMESPACE
oc patch route $APP_NAME --type=merge -p '{"spec":{"tls":{"termination":"edge"}}}' --namespace $NAMESPACE

echo ">> Starting build for $APP_NAME.."
oc start-build $APP_NAME --from-dir=. --namespace=$NAMESPACE --follow

echo ">> Adding (cluster-admin) permissions to the service account default in namespace $NAMESPACE.."
# !IMPORTANT: This is a cluster-admin role, please use with caution
oc adm policy add-cluster-role-to-user cluster-admin -z default -n $NAMESPACE

echo ">> remember to deploy Kueue CRDS"
