import httpx
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


async def send_slack_notification(message: str):
    """
    Sends a notification to a Slack channel via a webhook.
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL not set, skipping notification.")
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.SLACK_WEBHOOK_URL,
                json={"text": message},
                timeout=10.0,
            )
            response.raise_for_status()
            logger.debug(f"Slack notification sent successfully: {message}")
    except httpx.RequestError as e:
        logger.error(f"Error sending Slack notification: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while sending a Slack notification: {e}"
        )
