#!/bin/bash

APP_NAME="finetune-controller"

# Try to read NAMESPACE from .env file, default to "default" if not found
if [ -f .env ]; then
    NAMESPACE=$(grep '^NAMESPACE=' .env | cut -d '=' -f2 || echo "default")
else
    NAMESPACE="default"
fi

echo "Found namespace from .env: $NAMESPACE"
read -p "Press Enter to continue with this namespace, or type a new one: " input_namespace

if [ ! -z "$input_namespace" ]; then
    NAMESPACE=$input_namespace
fi

echo ">> Using namespace: $NAMESPACE"

echo ">> Archiving git project and uploading to cluster for build.."
git archive --format=tar HEAD | gzip > source.tar.gz

oc start-build $APP_NAME --from-archive=source.tar.gz --namespace=$NAMESPACE --follow

echo ">> cleaning up.."
rm -rf source.tar.gz
