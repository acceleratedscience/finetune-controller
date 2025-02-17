import logging
from app.database.db import db_manager
from app.utils.S3Handler import s3_handler
from app.schemas.db_schemas import PromotionStatus


logger = logging.getLogger(__name__)


class PromotionTask:
    async def promote_job_task(
        job_id: str,
        atrifacts_uri: str,
        destination_uri: str,
    ) -> None:
        """Background task to handle job promotion"""
        try:
            # Update status to in progress
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.IN_PROGRESS, destination_uri
            )

            # Copy S3 objects
            await s3_handler.copy_s3_object(atrifacts_uri, destination_uri)

            # Update database that job has been promoted
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.COMPLETED, destination_uri
            )
            logger.info(f"promotion complete for job {job_id}")

        except Exception as e:
            logger.error(f"Promotion failed for job {job_id}: {str(e)}")
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.FAILED, destination_uri
            )

    async def unpromote_job_task(
        job_id: str,
        destination_uri: str,
    ) -> None:
        """Background task to handle job promotion"""
        try:
            # Update status to deleting
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.DELETING, destination_uri
            )
            # Cleanup S3 objects
            await s3_handler.cleanup_uri_items(destination_uri)

            # Update database that job has been unpromoted
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.NOT_PROMOTED, None
            )
            logger.info(f"unpromotion complete for job {job_id}")

        except Exception as e:
            logger.error(f"Unpromotion failed for job {job_id}: {str(e)}")
            # keep state as promoted
            await db_manager.update_job_promotion(
                job_id, PromotionStatus.COMPLETED, destination_uri
            )
