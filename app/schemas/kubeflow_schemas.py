import logging
from enum import Enum

from app.schemas.db_schemas import DatabaseStatusEnum


logger = logging.getLogger(__name__)


class KubeflowStatusEnum(str, Enum):
    """kubeflow job states"""

    # JobCreated means the job has been accepted by the system,
    # but one or more of the pods/services has not been started.
    # This includes time before pods being scheduled and launched.
    created = "Created"
    # JobRunning means all sub-resources (e.g. services/pods) of this job
    # have been successfully scheduled and launched.
    # The training is running without error.
    running = "Running"
    # JobRestarting means one or more sub-resources (e.g. services/pods) of this job
    # reached phase failed but maybe restarted according to it's restart policy
    # which specified by user in v1.PodTemplateSpec.
    # The training is freezing/pending.
    restarting = "Restarting"
    # JobSucceeded means all sub-resources (e.g. services/pods) of this job
    # reached phase have terminated in success.
    # The training is complete without error.
    succeeded = "Succeeded"
    # JobSuspended means the job has been suspended.
    suspended = "Suspended"  # kueue state
    # JobFailed means one or more sub-resources (e.g. services/pods) of this job
    # reached phase failed with no restarting.
    # The training has failed its execution.
    failed = "Failed"


class TrainingJobStatus:
    """Helper class to check state of job"""

    # helpers to help check database state or kubeflow states
    running_states: list[DatabaseStatusEnum] = [
        DatabaseStatusEnum.queued,
        DatabaseStatusEnum.starting,
        DatabaseStatusEnum.running,
        DatabaseStatusEnum.restarting,
        KubeflowStatusEnum.created,
        KubeflowStatusEnum.running,
        KubeflowStatusEnum.suspended,
        KubeflowStatusEnum.restarting,
    ]
    stopped_states: list[DatabaseStatusEnum] = [
        DatabaseStatusEnum.completed,
        DatabaseStatusEnum.failed,
        DatabaseStatusEnum.canceled,
        DatabaseStatusEnum.error,
        KubeflowStatusEnum.succeeded,
        KubeflowStatusEnum.failed,
    ]

    @classmethod
    def map_status(
        cls, kubeflow_status: KubeflowStatusEnum | str
    ) -> DatabaseStatusEnum:
        """Maps a Kubeflow job status to Database status.

        Args:
            kubeflow_status (KubeflowStatusEnum): kubeflow status

        Returns:
            DatabaseStatusEnum: database status
        """
        try:
            status_mapping = {
                KubeflowStatusEnum.suspended: DatabaseStatusEnum.queued,
                KubeflowStatusEnum.created: DatabaseStatusEnum.starting,
                KubeflowStatusEnum.running: DatabaseStatusEnum.running,
                KubeflowStatusEnum.restarting: DatabaseStatusEnum.restarting,
                KubeflowStatusEnum.succeeded: DatabaseStatusEnum.completed,
                KubeflowStatusEnum.failed: DatabaseStatusEnum.failed,
            }
            return status_mapping[kubeflow_status]
        except KeyError as e:
            logging.error(f"job in unkown state: {kubeflow_status}")
            return DatabaseStatusEnum.error


if __name__ == "__main__":
    # Mapping a Kubeflow status to a database status
    kubeflow_status = [KubeflowStatusEnum.created, KubeflowStatusEnum.failed]

    for status in kubeflow_status:
        database_status = TrainingJobStatus.map_status(status)

        print(f"Kubeflow Status: {status}, Database Status: {database_status}")

        if database_status in TrainingJobStatus.running_states:
            print("Job is running")
        elif database_status in TrainingJobStatus.stopped_states:
            print("Job has stopped")
