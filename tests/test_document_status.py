import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.services.storage.fake_store import FakeFileStore
from app.services.storage.s3_store import get_file_store


@pytest.fixture
def store():
    fake = FakeFileStore()
    app.dependency_overrides[get_file_store] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_file_store, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def make_document(db_session, seeded_user):
    """A document owned by the logged-in user, cleaned up afterwards."""
    created: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(**fields) -> Document:
        user = db_session.execute(select(User).where(User.email == email)).scalars().one()
        document = Document(
            id=uuid.uuid4(),
            user_id=user.id,
            source=fields.pop("source", DocumentSource.PDF),
            status=fields.pop("status", IngestionStatus.PENDING),
            content=fields.pop("content", "Razom 349.50 UAH, ATB Market, chek 10482"),
            title=fields.pop("title", "receipt"),
            **fields,
        )
        db_session.add(document)
        db_session.commit()
        created.append(document.id)
        return document

    yield build

    for document_id in created:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()


def test_a_pending_document_reports_pending(client, auth_headers, make_document):
    document = make_document()

    body = client.get(f"/api/v1/documents/{document.id}", headers=auth_headers).json()

    assert body["status"] == IngestionStatus.PENDING
    assert body["chunk_count"] == 0
    assert body["error_message"] is None
    assert body["source"] == DocumentSource.PDF


def test_a_failed_document_reports_why(client, auth_headers, make_document):
    """
    The whole reason this endpoint exists. error_message is written by the
    pipeline and was previously unreachable from outside the database, so
    a user whose scan failed got silence.
    """
    document = make_document(
        status=IngestionStatus.FAILED,
        error_message="Extracted text too short to be useful (0 chars)",
    )

    body = client.get(f"/api/v1/documents/{document.id}", headers=auth_headers).json()

    assert body["status"] == IngestionStatus.FAILED
    assert "too short" in body["error_message"]
    assert body["chunk_count"] == 0


def test_chunk_count_is_what_actually_answers_searchable(
        client, auth_headers, make_document, db_session
):
    """
    Status and chunk_count can disagree, and chunk_count is the
    one that matters: zero chunks means nothing reached the index
    """
    document = make_document(status=IngestionStatus.COMPLETED)
    db_session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            content="Razom 349.50",
            start_char=0,
            end_char=12,
        )
    )
    db_session.commit()

    body = client.get(f"/api/v1/documents/{document.id}", headers=auth_headers).json()

    assert body["status"] == IngestionStatus.COMPLETED
    assert body["chunk_count"] == 1


def test_file_metadata_is_reported(client, auth_headers, make_document):
    document = make_document(
        source=DocumentSource.IMAGE,
        storage_key="documents/x/y.png",
        file_size=1234,
        page_map=[{"index": 0, "start_char": 0, "end_char": 10, "needs_ocr": False}],
    )

    body = client.get(f"/api/v1/documents/{document.id}", headers=auth_headers).json()

    assert body["has_file"] is True
    assert body["page_count"] == 1
    assert body["source"] == DocumentSource.IMAGE


def test_a_document_without_a_file_is_fine(client, auth_headers, make_document):
    """Articles and selections have no stored bytes and must still report."""
    body = client.get(
        f"/api/v1/documents/{make_document(source=DocumentSource.ARTICLE).id}",
        headers=auth_headers,
    ).json()

    assert body["has_file"] is False
    assert body["page_count"] == 0


def test_a_missing_document_is_404(client, auth_headers):
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}", headers=auth_headers)
    assert response.status_code == 404


def test_another_users_document_is_404_not_403(
        client, auth_headers, db_session, make_document
):
    """
    Same code as "does not exist", deliberately. A 403 would confirm the
    id is real, which is an enumeration oracle over other people's
    receipts — the same reasoning as GET /documents/{id}/file.
    """
    document = make_document()
    stranger = User(id=uuid.uuid4(), email=f"stranger-{uuid.uuid4().hex[:8]}@example.com")
    db_session.add(stranger)
    db_session.commit()
    document.user_id = stranger.id
    db_session.commit()

    try:
        response = client.get(f"/api/v1/documents/{document.id}", headers=auth_headers)
        assert response.status_code == 404
        assert "No such document" in response.json()["error"]["message"]
    finally:
        db_session.delete(document)
        db_session.commit()
        db_session.delete(stranger)
        db_session.commit()


def test_authentication_is_required(client, make_document):
    response = client.get(f"/api/v1/documents/{make_document().id}")
    assert response.status_code in (401, 403)