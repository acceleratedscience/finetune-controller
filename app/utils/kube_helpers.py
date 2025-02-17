import logging

from app.utils.kube_config import api_instance, core_v1_api

# Create a logger
logger = logging.getLogger(__name__)


def check_status(job_id, namespace):
    # job_status = api_instance.get_namespaced_custom_object(
    job_status = api_instance.get_namespaced_custom_object_status(
        group="kubeflow.org",
        version="v1",
        namespace=namespace,
        plural="pytorchjobs",
        name=job_id,
    )
    return job_status


def get_pod_by_selector(selector, namespace):
    pods = core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=selector)
    return pods.items[0].metadata.name


async def get_pod_events(selector, namespace):
    pod_name = get_pod_by_selector(selector, namespace)
    # List all events in the specified namespace
    req = core_v1_api.list_event_for_all_namespaces(async_req=True)
    # events = core_v1_api.list_namespaced_event(namespace=namespace)
    # Filter events related to the specified pod
    events = req.get()
    pod_events = [
        event for event in events.items if event.involved_object.name == pod_name
    ]
    # Print event details
    for event in pod_events:
        print(f"Event: {event.message} | Type: {event.type} | Reason: {event.reason}")
    return pod_events


async def get_pod_status(selector, namespace):
    """return a dictionary if pod is failing"""
    try:
        pod_name = get_pod_by_selector(selector, namespace)
        # Get the pod details in the specified namespace
        req = core_v1_api.read_namespaced_pod_status(
            name=pod_name, namespace=namespace, async_req=True
        )
        pod_status = req.get()
        # Print the pod status details
        # print(f"Pod Name: {pod_status.metadata.name}")
        # print(f"Namespace: {pod_status.metadata.namespace}")
        # print(f"Phase: {pod_status.status.phase}")
        # print(f"Conditions: {pod_status.status.conditions}")
        # print(f"Host IP: {pod_status.status.host_ip}")
        # print(f"Pod IP: {pod_status.status.pod_ip}")
        # print(f"Start Time: {pod_status.status.start_time}")
        if not pod_status.status.container_statuses:
            return None

        status_info = {
            "restart_count": 0,
            "start_time": (
                pod_status.status.start_time.isoformat()
                if pod_status.status.start_time
                else None
            ),
        }

        # Add completion time and elapsed time if container has terminated
        for container_status in pod_status.status.container_statuses:
            status_info["restart_count"] += container_status.restart_count
            if container_status.state.terminated:
                finished_at = container_status.state.terminated.finished_at
                status_info["completion_time"] = finished_at.isoformat()

                # Calculate elapsed time in seconds if we have both start and end times
                if pod_status.status.start_time:
                    elapsed = (
                        finished_at - pod_status.status.start_time
                    ).total_seconds()
                    status_info["elapsed_time"] = f"{elapsed:.1f}s"
            elif container_status.state.waiting:
                status_info.update(
                    {
                        "message": container_status.state.waiting.message,
                        "reason": container_status.state.waiting.reason,
                        "status": container_status.ready,
                    }
                )
        return status_info
    except Exception as e:
        logger.error(str(e))
        return None
