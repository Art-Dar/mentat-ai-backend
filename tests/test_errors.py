import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.errors import (
    REQUEST_ID_HEADER,
    Conflict,
    NotFound,
    Unauthorized,
    register_error_handling,
)


class Payload(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    register_error_handling(app)

    @app.post("/echo")
    async def echo(payload: Payload):
        return {"ok": True}

    @app.get("/not-found")
    async def not_found():
        raise NotFound("Document not found")

    @app.get("/conflict")
    async def conflict():
        raise Conflict()

    @app.get("/unauthorized")
    async def unauthorized():
        raise Unauthorized("Token expired")

    @app.get("/http-exception")
    async def http_exception():
        raise HTTPException(status_code=403, detail="Nope")

    @app.get("/integrity")
    async def integrity():
        raise IntegrityError(
            "INSERT INTO users (email) VALUES ($1)",
            {"email": "a@b.c"},
            Exception('duplicate key value violates unique constraint "ix_users_email"'),
        )

    @app.get("/db-down")
    async def db_down():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret connection string postgresql://user:pw@host/db")

    # raise_server_exceptions=False so the 500 handler's response is returned
    # instead of the exception being re-raised into the test.
    return TestClient(app, raise_server_exceptions=False)


# envelope shape


def test_domain_error_uses_the_envelope(client):
    response = client.get("/not-found")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["message"] == "Document not found"
    assert error["request_id"]


def test_default_message_is_used_when_none_given(client):
    error = client.get("/conflict").json()["error"]
    assert error["code"] == "conflict"
    assert error["message"] == "Resource already exists"


def test_details_omitted_when_empty(client):
    assert "details" not in client.get("/not-found").json()["error"]


def test_fastapi_http_exception_is_wrapped_too(client):
    response = client.get("/http-exception")

    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "forbidden"
    assert error["message"] == "Nope"


def test_status_codes_are_preserved(client):
    assert client.get("/unauthorized").status_code == 401
    assert client.get("/conflict").status_code == 409


# request id


def test_request_id_is_returned_in_header_and_body(client):
    response = client.get("/not-found")
    assert response.headers[REQUEST_ID_HEADER]
    assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_incoming_request_id_is_preserved(client):
    response = client.get("/not-found", headers={REQUEST_ID_HEADER: "caller-supplied-id"})
    assert response.headers[REQUEST_ID_HEADER] == "caller-supplied-id"
    assert response.json()["error"]["request_id"] == "caller-supplied-id"


def test_successful_responses_also_carry_a_request_id(client):
    response = client.post("/echo", json={"email": "a@b.c", "password": "long-enough-password"})
    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


# validation


def test_validation_error_lists_offending_fields(client):
    response = client.post("/echo", json={"email": "x", "password": "short"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    fields = {detail["field"] for detail in error["details"]}
    assert fields == {"body.email", "body.password"}


def test_validation_error_does_not_echo_submitted_values(client):
    """Pydantic includes the rejected input by default; we must strip it."""
    secret = "hunter2-this-must-not-appear"
    response = client.post("/echo", json={"email": "x", "password": secret})

    assert secret not in response.text


def test_missing_body_is_a_validation_error(client):
    response = client.post("/echo", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# database


def test_integrity_error_becomes_409(client):
    response = client.get("/integrity")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_integrity_error_does_not_leak_schema(client):
    response = client.get("/integrity")
    assert "ix_users_email" not in response.text
    assert "unique constraint" not in response.text


def test_database_outage_becomes_503(client):
    response = client.get("/db-down")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"
    assert "connection refused" not in response.text


# unhandled


def test_unhandled_exception_becomes_500_envelope(client):
    response = client.get("/boom")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert error["request_id"]


def test_unhandled_exception_does_not_leak_its_message(client):
    response = client.get("/boom")
    assert "postgresql://" not in response.text
    assert "secret connection string" not in response.text


def test_http_exception_headers_are_preserved(client):
    """OAuth2PasswordBearer's WWW-Authenticate challenge must survive."""
    from fastapi import FastAPI
    from fastapi.security import OAuth2PasswordBearer
    from fastapi.testclient import TestClient as TC

    from app.core.errors import register_error_handling

    app = FastAPI()
    register_error_handling(app)
    scheme = OAuth2PasswordBearer(tokenUrl="/token")

    @app.get("/guarded")
    async def guarded(token: str = Depends(scheme)):
        return {"ok": True}

    response = TC(app, raise_server_exceptions=False).get("/guarded")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"
