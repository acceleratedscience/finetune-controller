import logging
from app.core.security import OpenBridgeBasicMiddleware
from app.core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi import Limiter


from fastapi import FastAPI


# Create a logger
logger = logging.getLogger(__name__)


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def openbridge_middleware(app: FastAPI) -> None:
    # Add OpenBridge middleware
    if settings.ENVIRONMENT != "local" and (
        not settings.OPENBRIDGE_INTROSPECTION_URL
        or not settings.OPENBRIDGE_CLIENT_ID
        or not settings.OPENBRIDGE_CLIENT_SECRET
    ):
        logger.critical(
            "OpenBridge middleware not configured. Your API routes are exposed!"
        )
    elif not settings.OPENBRIDGE_INTROSPECTION_URL or not (
        settings.OPENBRIDGE_CLIENT_ID and settings.OPENBRIDGE_CLIENT_SECRET
    ):
        logger.warning(
            "OpenBridge middleware not configured. Missing introspection URL or API key"
        )
    elif (
        settings.ENVIRONMENT in ["local", "staging"]
        and settings.DEV_DISABLE_INTROSPECTION
    ):
        # Disable introspection validation for local and staging development only
        logger.warning("OpenBridge middleware Introspection disabled for development")
    else:
        logger.info("Configuring OpenBridge Basic Auth middleware")
        app.add_middleware(OpenBridgeBasicMiddleware)


def cors_middleware(app: FastAPI) -> None:
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.FRONTEND_URL_CORS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_middleware(app: FastAPI) -> None:
    # Add OpenBridge middleware
    openbridge_middleware(app)
    # CORS Middleware
    cors_middleware(app)
    # Initialize rate limiter middlware
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
