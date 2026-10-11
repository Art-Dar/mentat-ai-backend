# /ingest all the way through to stored embeddings
# HTTP request -> document row -> BackgroundTask ->
# normalize -> chunk -> embed -> chunks in Postgres

# Starlette's TestClient runs background tasks before it returns

import uuid

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import select

from app.models import Chunk, Document, IngestionStatus
from app.models.chunk import EMBEDDING_DIM

ARTICLE = "\n\n".join(
    [
        "Нормалізація тексту виконується перед розбиттям на фрагменти. Вона "
        "приводить пробіли до єдиного вигляду, розкодовує HTML-сутності та "
        "застосовує форму NFKC до символів Юнікоду. Без цього однакові за "
        "змістом тексти дають різні вектори, і дублікати неможливо виявити.",
        "Vector databases store high-dimensional embeddings and retrieve them "
        "by approximate nearest neighbour search. The HNSW index builds a "
        "navigable small-world graph over the vectors, so recall stays high "
        "without scanning every row in the table at query time.",
    ]
)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def captured(client, auth_headers, db_session):
    """POST a capture and clean up the document afterwards."""
    created: list[uuid.UUID] = []

    def post(text: str = ARTICLE, **extra):
        response = client.post(
            "/api/v1/ingest",
            headers=auth_headers,
            json={"source": "article", "text": text, **extra},
        )
        assert response.status_code == 201
        document_id = uuid.UUID(response.json()["document_id"])
        created.append(document_id)
        return response, document_id

    yield post

    for document_id in created:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()


@pytest.mark.runs_pipeline
def test_ingest_responds_before_the_pipeline_finishes(captured):
    """
    The response is built and sent first; the task runs after. So the body
    says PENDING even though the row is COMPLETED moments later.
    """
    response, _ = captured()

    assert response.json()["status"] == "pending"


@pytest.mark.runs_pipeline
def test_the_document_reaches_completed(captured, db_session):
    _, document_id = captured()

    document = db_session.get(Document, document_id)
    db_session.refresh(document)
    assert document.status is IngestionStatus.COMPLETED
    assert document.error_message is None


@pytest.mark.runs_pipeline
def test_chunks_and_embeddings_are_stored(captured, db_session):
    """/ingest now produces searchable rows."""
    _, document_id = captured()

    chunks = db_session.scalars(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    ).all()

    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(len(chunk.embedding) == EMBEDDING_DIM for chunk in chunks)


@pytest.mark.runs_pipeline
def test_the_capture_is_normalized_in_place(captured, db_session):
    """
    `content` holds the raw capture until the pipeline runs, then holds the
    normalized text. Any test asserting content == the posted string is
    asserting that the pipeline did NOT run.
    """
    _, document_id = captured()

    document = db_session.get(Document, document_id)
    db_session.refresh(document)
    assert document.language in {"uk", "en"}
    assert document.content_hash and len(document.content_hash) == 64


@pytest.mark.runs_pipeline
def test_a_capture_too_short_to_use_is_recorded_as_failed(captured, db_session):
    """
    A failure in a background task must land on the row. Nothing awaits
    that coroutine, so an exception there is swallowed — the document would
    otherwise sit at PROCESSING with no explanation.
    """
    _, document_id = captured(text="Too short.")

    document = db_session.get(Document, document_id)
    db_session.refresh(document)
    assert document.status is IngestionStatus.FAILED
    assert "too short" in document.error_message.lower()


def test_by_default_endpoint_tests_do_not_run_the_pipeline(client, auth_headers, db_session):
    """
    The guard itself. Without the conftest fixture, every /ingest test would
    run a full pipeline and reach for the Voyage API from CI.
    """
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "article", "text": ARTICLE},
    )
    document_id = uuid.UUID(response.json()["document_id"])

    document = db_session.get(Document, document_id)
    db_session.refresh(document)
    assert document.status is IngestionStatus.PENDING
    assert document.content == ARTICLE  # untouched: nothing normalized it

    db_session.delete(document)
    db_session.commit()


def test_the_endpoint_hands_off_instead_of_awaiting(
    client, auth_headers, db_session, pipeline_isolation
):
    """
    responds immediately instead of blocking on the full pipeline

    Awaiting process_document here would still produce a correct document
    and a 201 saying "pending" — every other assertion in this file would
    pass. What distinguishes the two is that the endpoint hands a real
    BackgroundTasks instance to the scheduler.
    """
    response = client.post(
        "/api/v1/ingest",
        headers=auth_headers,
        json={"source": "article", "text": ARTICLE},
    )
    document_id = uuid.UUID(response.json()["document_id"])

    assert len(pipeline_isolation) == 1
    scheduled_id, background_tasks = pipeline_isolation[0]
    assert scheduled_id == document_id
    assert isinstance(background_tasks, BackgroundTasks)

    db_session.delete(db_session.get(Document, document_id))
    db_session.commit()
