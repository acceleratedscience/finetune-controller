import httpx
from jose import jwt, JWTError
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
from app.core.config import settings
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel


# Create a logger
logger = logging.getLogger(__name__)


class OpenBridgeBasicMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate OpenBridge tokens via introspection.

    Authorization header is Basic Auth with client_id and client_secret
    """

    async def dispatch(self, request: Request, call_next):
        try:
            # Only validate for routes starting with `/api/v1/`
            if request.url.path.startswith(settings.API_V1_STR):
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401,
                        detail="Missing or invalid Authorization header",
                    )

                token = auth_header.split(" ")[1]

                # Create Basic Auth header
                client_id = settings.OPENBRIDGE_CLIENT_ID
                client_secret = settings.OPENBRIDGE_CLIENT_SECRET.get_secret_value()
                authorization = httpx.BasicAuth(client_id, client_secret)
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",  # Required for OAuth2
                }
                # users JWT token for introspection
                data = {"token": token}

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        settings.OPENBRIDGE_INTROSPECTION_URL,
                        headers=headers,
                        auth=authorization,
                        data=data,  # Use `data` instead of `json` for OAuth2 form data
                    )
                if response.status_code != 200 or not response.json().get("active"):
                    raise HTTPException(
                        status_code=401, detail="Token validation failed"
                    )

                # Store the token data in request.state
                request.state.jwt_data = response.json()

            return await call_next(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "status_code": exc.status_code},
            )
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return JSONResponse(
                status_code=500, content={"detail": "Something went wrong"}
            )


class OpenBridgeJWTMiddleware(BaseHTTPMiddleware):
    """ "
    Middleware to validate OpenBridge tokens via introspection.

    Authorization header is JWT token
    """

    async def dispatch(self, request: Request, call_next):
        try:
            # Only validate for routes starting with `/api/v1/`
            if request.url.path.startswith(settings.API_V1_STR):
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401,
                        detail="Missing or invalid Authorization header",
                    )

                token = auth_header.split(" ")[1]
                headers = {
                    "Authorization": f"Bearer {settings.OPENBRIDGE_API_KEY.get_secret_value()}",
                    "Content-Type": "application/x-www-form-urlencoded",  # Required for OAuth2
                }
                # users JWT token for introspection
                data = {"token": token}

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        settings.OPENBRIDGE_INTROSPECTION_URL,
                        headers=headers,
                        data=data,  # Use `data` instead of `json` for OAuth2 form data
                    )

                if response.status_code != 200 or not response.json().get("active"):
                    raise HTTPException(
                        status_code=401, detail="Token validation failed"
                    )

            return await call_next(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail, "status_code": exc.status_code},
            )
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return JSONResponse(
                status_code=500, content={"error": "Something went wrong"}
            )


class UserJWT(BaseModel):
    user_id: str
    audience: str
    available_models: list[str]
    expires: int


def decode_jwt(token: str) -> dict:
    try:
        # Decode and verify the token
        payload = jwt.get_unverified_claims(
            token,
        )
        return UserJWT(
            user_id=payload.get("sub"),
            audience=payload.get("aud"),
            available_models=payload.get("scp", []),
            expires=payload.get("exp"),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


auth_scheme = HTTPBearer(auto_error=None)


def decode_jwt_header(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),
) -> UserJWT | None:
    if credentials:
        token = credentials.credentials
        return decode_jwt(token)
    else:
        return None


### Commented out code below is for reference only for Bearer token authorization ###

# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from fastapi import Depends, Security

# auth_scheme = HTTPBearer()


# async def introspect_token(
#     credentials: HTTPAuthorizationCredentials = Security(auth_scheme),
# ) -> dict:
#     token = credentials.credentials
#     headers = {
#         "Authorization": f"Bearer {settings.OPENBRIDGE_API_KEY.get_secret_value()}"
#     }
#     data = {"token": token}

#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             settings.OPENBRIDGE_INTROSPECTION_URL, headers=headers, json=data
#         )

#     if response.status_code != 200:
#         raise HTTPException(
#             status_code=401,
#             detail="Token validation failed",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     token_data = response.json()

#     # Optional: Ensure token is active and has required claims
#     if not token_data.get("active"):
#         raise HTTPException(
#             status_code=401,
#             detail="Inactive or invalid token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     return token_data


# # Conditionally set dependencies, only for local development
# # method for per route dependency injection
# introspect_dependencies = (
#     []
#     if settings.ENVIRONMENT == "local" and settings.DEV_DISABLE_INTROSPECTION
#     else [Depends(introspect_token)]
# )
