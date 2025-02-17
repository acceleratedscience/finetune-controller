from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from app.core.config import settings


def custom_openapi_jwt_auth(app: FastAPI):
    """Customize OpenAPI schema to use JWT Authentication"""

    def custom_api():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="Finetune Controller API",
            version="1.0.0",
            description="API with JWT Authentication",
            routes=app.routes,
        )
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
        }
        # Apply security scheme to `/api/v1/` routes
        for path, methods in openapi_schema["paths"].items():
            if path.startswith(settings.API_V1_STR):
                for method in methods:
                    methods[method]["security"] = [{"BearerAuth": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    return custom_api


def custom_openapi_basic_auth(app: FastAPI):
    """Customize OpenAPI schema to use Basic Authentication"""

    def custom_api():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="Finetune Controller API",
            version="1.0.0",
            description="API with Basic Authentication",
            routes=app.routes,
        )

        # Define Basic Authentication in OpenAPI
        openapi_schema["components"]["securitySchemes"] = {
            "BasicAuth": {"type": "http", "scheme": "basic"}
        }

        # Apply security scheme to `/api/v1/` routes
        for path, methods in openapi_schema["paths"].items():
            if path.startswith(settings.API_V1_STR):
                for method in methods:
                    methods[method]["security"] = [{"BasicAuth": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    return custom_api
