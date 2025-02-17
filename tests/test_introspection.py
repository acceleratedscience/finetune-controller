from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta


class IntrospectionRequest(BaseModel):
    token: str


app = FastAPI()

# Store valid tokens with their metadata
valid_tokens = {
    "valid_token": {
        "active": True,
        "scope": "read write",
        "client_id": "test_client",
        "username": "test_user",
        "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp(),
    },
    "expired_token": {
        "active": False,
        "scope": "read write",
        "client_id": "test_client",
        "username": "test_user",
        "exp": (datetime.utcnow() - timedelta(hours=1)).timestamp(),
    },
}


@app.post("/oauth/introspect")
async def introspect_json(request: IntrospectionRequest = Form(...)):
    # Check Authorization header in a real implementation

    if request.token in valid_tokens:
        return valid_tokens[request.token]

    return {"active": False}


@app.get("/test/protected")
async def protected_route():
    return {"message": "You accessed a protected route"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
