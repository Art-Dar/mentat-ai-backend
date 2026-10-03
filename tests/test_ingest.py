import uuid

import pytest
from sqlalchemy import select

from app.models import Document, IngestionStatus

LONG_TEXT = (
    "Сьогодні в Києві відбулася важлива зустріч представників різних галузей "
    "економіки для обговорення подальшого розвитку інфраструктури міста."
)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _get_document(db_session, document_id):
    return db_session.scalar(select(Document).where(Document.id == document_id))


def test_ingest_creates_document_and_returns_id(client, db_session, auth_headers):
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={
            "source": "article",
            "text": LONG_TEXT,
            "url": "https://example.com/article",
            "title": "Зустріч у Києві",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"

    document = _get_document(db_session, uuid.UUID(body["document_id"]))
    assert document is not None
    assert document.status is IngestionStatus.PENDING
    assert document.url == "https://example.com/article"
    assert document.title == "Зустріч у Києві"
    assert document.content == LONG_TEXT  # raw capture, not yet normalized


def test_ingest_attributes_document_to_the_caller(client, db_session, auth_headers, seeded_user):
    email, _ = seeded_user

    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "note", "text": LONG_TEXT},
    )

    document = _get_document(db_session, uuid.UUID(response.json()["document_id"]))
    assert document.user.email == email


def test_ingest_accepts_capture_without_url(client, auth_headers):
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "note", "text": LONG_TEXT},
    )
    assert response.status_code == 201


def test_ingest_requires_auth(client):
    response = client.post(
        "/api/v1/ingest",
        json={"source": "article", "text": LONG_TEXT},
    )
    assert response.status_code in (401, 403)


def test_ingest_rejects_unsupported_source(client, auth_headers):
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "image", "text": LONG_TEXT},
    )
    assert response.status_code == 422


def test_ingest_rejects_empty_text(client, auth_headers):
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "note", "text": ""},
    )
    assert response.status_code == 422


def test_ingest_rejects_malformed_url(client, auth_headers):
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "article", "text": LONG_TEXT, "url": "not-a-url"},
    )
    assert response.status_code == 422
