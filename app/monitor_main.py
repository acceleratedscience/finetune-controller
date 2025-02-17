import asyncio
import signal
import sys
import logging
from contextlib import AsyncExitStack

from app.utils.kube_config import load_k8s_config
from app.database.db import db_manager
from app.core.monitor import job_monitor
from app.utils.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

load_k8s_config()


async def shutdown(signal, loop, exit_stack):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {signal.name}...")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    for task in tasks:
        task.cancel()

    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)

    await exit_stack.aclose()
    loop.stop()


def handle_exception(loop, context):
    """Handle exceptions outside of coroutines"""
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}")
    logger.info("Shutting down...")
    asyncio.create_task(shutdown(signal.SIGTERM, loop, exit_stack))


async def main():
    # Store exit_stack globally for shutdown handler
    global exit_stack
    async with AsyncExitStack() as exit_stack:
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Add signal handlers
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s, loop, exit_stack))
            )

        # Set up exception handler
        loop.set_exception_handler(handle_exception)

        try:
            # Startup: connect to the database
            await db_manager.connect()
            # Start the job monitor
            await job_monitor.start()

            # Keep the service running
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Main task cancelled")
        except Exception as e:
            logger.exception(f"Unexpected error in main: {e}")
            raise
        finally:
            # Ensure monitor is stopped
            await job_monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Service stopped")
