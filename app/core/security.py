import logging
import time
import json
import httpx
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel
from jose import jwt, jwk, JWTError, ExpiredSignatureError
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi import Security

from app.core.config import settings
from app.jobs.registered_models import JOB_MANIFESTS


# Create a logger
logger = logging.getLogger(__name__)


# Create a Pydantic model to match your cookie structure
class CookieData(BaseModel):
    subject: str
    user_type: str
    config: dict | None = None
    resources: list[str]
    token: str


class UserJWT(BaseModel):
    user_id: str
    audience: str
    available_models: list[str]
    expires: int


class JWTExtraInfo(BaseModel):
    active: bool
    user_id: str
    group_id: str


def decode_jwt(token: str) -> UserJWT:
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


class TokenValidator:
    def __init__(
        self,
        jwks_url: str,
        introspection_url: str,
        client_id: str,
        client_secret: str,
        jwks_cache_time: int = 300,  # 5 minutes
        metadata_cache_time: int = 60,  # 1 minute
    ):
        self.jwks_url = jwks_url
        self.introspection_url = introspection_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.jwks_cache_time = jwks_cache_time
        self.metadata_cache_time = metadata_cache_time
        self._jwks_cache = None
        self._jwks_timestamp = 0
        # Cache user metadata using sub as key
        self._metadata_cache = {}
        self._metadata_timestamps = {}

    async def _fetch_jwks(self) -> dict:
        """Fetch and cache JWKS"""
        current_time = time.time()
        if (
            self._jwks_cache is None
            or current_time - self._jwks_timestamp > self.jwks_cache_time
        ):
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                self._jwks_cache = response.json()
                self._jwks_timestamp = current_time
        return self._jwks_cache

    async def _get_public_key(self, token: str) -> str | None:
        """Get public key from JWKS"""
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                return None

            jwks = await self._fetch_jwks()
            for key in jwks["keys"]:
                if key["kid"] == kid:
                    return jwk.construct(key).to_pem()
            return None
        except Exception:
            return None

    async def _introspect_token(self, token: str) -> dict:
        """Introspect token at the authorization server"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.introspection_url,
                data={
                    "token": token,
                },
                auth=httpx.BasicAuth(self.client_id, self.client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return response.json()

    def _is_metadata_cached(self, sub: str) -> bool:
        """Check if user metadata is cached and valid"""
        if sub in self._metadata_cache:
            timestamp = self._metadata_timestamps.get(sub, 0)
            if time.time() - timestamp <= self.metadata_cache_time:
                return True
            else:
                # Clean up expired cache entry
                del self._metadata_cache[sub]
                del self._metadata_timestamps[sub]
        return False

    def _cache_metadata(self, metadata: dict):
        """Cache user metadata using sub as key"""
        if "sub" in metadata:
            self._metadata_cache[metadata["sub"]] = metadata
            self._metadata_timestamps[metadata["sub"]] = time.time()

    async def validate_token(self, token: str) -> dict:
        """Validate token using JWK first, then fallback to introspection"""
        try:
            # Try JWK validation first
            public_key = await self._get_public_key(token)
            if public_key:
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=["ES256"],
                    options={
                        "verify_exp": True,
                        "verify_aud": False,
                    },
                )
                # If we have cached metadata for this user, return it
                if "sub" in payload and self._is_metadata_cached(payload["sub"]):
                    return self._metadata_cache[payload["sub"]]

                # If JWK validation succeeds but we need metadata,
                # still do introspection
                introspection_result = await self._introspect_token(token)
                if not introspection_result.get("active", False):
                    raise HTTPException(status_code=401, detail="Invalid token")

                self._cache_metadata(introspection_result)
                return introspection_result

        except JWTError:
            # If JWK validation fails, proceed to introspection
            pass

        # Fallback to introspection
        introspection_result = await self._introspect_token(token)

        if not introspection_result.get("active", False):
            raise HTTPException(status_code=401, detail="Invalid token")

        self._cache_metadata(introspection_result)
        return introspection_result


# Initialize validator
validator = TokenValidator(
    jwks_url=settings.OPENBRIDGE_JWK_URL,
    introspection_url=settings.OPENBRIDGE_INTROSPECTION_URL,
    client_id=settings.OPENBRIDGE_CLIENT_ID,
    client_secret=settings.OPENBRIDGE_CLIENT_SECRET.get_secret_value(),
)


class OpenBridgeBasicMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate OpenBridge tokens via introspection.

    Authorization header is Basic Auth with client_id and client_secret
    """

    async def dispatch(self, request: Request, call_next):
        try:
            # Only validate for routes starting with `/api/v1/`
            if request.url.path.startswith(settings.API_V1_STR):
                # TODO: local env jwt only, prod secure cookie
                # use secure cookie auth first
                # Get the cookie value
                cookie_value = request.cookies.get("bridge-user")
                if cookie_value:
                    # First decode it as JSON
                    cookie_json = json.loads(request.cookies.get("bridge-user"))
                    if cookie_json and cookie_json.get("token"):
                        # Parse into Pydantic model for validation
                        cookie_data = CookieData(**cookie_json)
                        token = cookie_data.token
                    else:
                        raise HTTPException(
                            status_code=401,
                            detail="Missing or invalid Authorization header",
                        )

                # use header auth second if no cookie available
                else:
                    auth_header = request.headers.get("Authorization")
                    if not auth_header or not auth_header.startswith("Bearer "):
                        raise HTTPException(
                            status_code=401,
                            detail="Missing or invalid Authorization header",
                        )
                    token = auth_header.split(" ")[1]

                try:
                    # validate the user
                    jwt_info = await validator.validate_token(token)
                except Exception as e:
                    # if failed check if mock jwt and local only
                    if settings.ENVIRONMENT == "local":
                        logger.warning("local enviornment using mock generated tokens!")
                        jwt_info = await dev_mock_token_introspection(token)
                    else:
                        raise e

                # Store the token data in request.state
                request.state.jwt_data = JWTExtraInfo(
                    active=jwt_info.get("active"),
                    user_id=jwt_info.get("sub"),
                    group_id=jwt_info.get("group_id"),
                )
                request.state.decoded_jwt = decode_jwt(token)

            return await call_next(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "status_code": exc.status_code},
            )
        except Exception as e:
            logger.error(f"Error validating token: {e}", exc_info=True)
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


auth_scheme = HTTPBearer(auto_error=None)


def decode_jwt_header(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),
) -> UserJWT | None:
    if credentials:
        token = credentials.credentials
        return decode_jwt(token)
    else:
        return None


def verify_secret_key():
    """Verify that the secret key is properly set."""
    if not settings.JWT_SECRET_KEY:
        logger.warning(
            "command to generate a secret: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
        raise HTTPException(
            status_code=500,
            detail="JWT_SECRET_KEY not configured. Please set it in your environment variables.",
        )


async def dev_generate_token(user: str, include_models: list[str] = None) -> dict:
    """Generate a JWT token with specific claims for development."""
    if include_models:
        # include only specific models
        user_models = include_models
    else:
        user_models = list(JOB_MANIFESTS.keys())
    try:
        # Verify secret key is configured
        verify_secret_key()

        # Create token expiration time
        expire = datetime.now(timezone.utc) + timedelta(hours=1)

        # Create claims
        jwt_claims = {
            "sub": user,
            "aud": "local",
            "scp": user_models,  # available models
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        # Generate token
        token = jwt.encode(
            jwt_claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        return token

    except Exception as e:
        logger.error(f"Failed to generate development access token\n{e}")
        # raise HTTPException(status_code=500, detail=f"Error generating token: {str(e)}")


# Optional: Add a verification endpoint
async def verify_token(token: str) -> dict:
    """Verify a JWT token and return its claims."""
    try:
        verify_secret_key()

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience="local",
        )
        return payload

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")


async def dev_mock_token_introspection(token: str) -> dict:
    """Mock introspection function for authentication testing"""
    payload = await verify_token(token)

    return {
        "active": True,
        "client_id": "client-id",
        "sub": payload.get("sub"),
        "group_id": "mock-group-id",
    }


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
