import logging

from kubernetes import client, config

# Create a logger
logger = logging.getLogger(__name__)


def load_k8s_config():
    # Load in-cluster config if running inside Kubernetes, else use local kubeconfig
    try:
        config.load_incluster_config()
        logger.info("loading kubernetes cluster config")
    except:
        logger.info("loading local kube config")
        config.load_kube_config()


load_k8s_config()

# Initialize Kubernetes clients
api_instance = client.CustomObjectsApi()
core_v1_api = client.CoreV1Api()
