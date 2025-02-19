from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.core.security import OpenBridgeBasicMiddleware

# Create a simple test app
app = FastAPI()
app.add_middleware(OpenBridgeBasicMiddleware)


@app.get("/test")
async def test_endpoint():
    return {"message": "success"}


client = TestClient(app)


def test_valid_token():
    response = client.get("/test", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert response.json() == {"message": "success"}


def test_invalid_token():
    response = client.get("/test", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401


def test_missing_token():
    response = client.get("/test")
    assert response.status_code == 401


def test_malformed_auth_header():
    response = client.get("/test", headers={"Authorization": "NotBearer token"})
    assert response.status_code == 401
