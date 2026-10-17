import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import get_embedder
from app.services.ingestion.fake_embedder import FakeEmbedder

PASSAGE = (
    "Espresso extraction depends on grind size, dose and water temperature. "
    "A shot pulled too fast tastes sour because the water has not had time "
    "to dissolve the sugars locked inside the puck."
)


@pytest.fixture
def fake_embedder():
    """No network, no Voyage bill, deterministic vectors."""
    fake = FakeEmbedder()
    app.dependency_overrides[get_embedder] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_embedder, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def make_chunk(db_session, seeded_user):
    """
    Written directly rather than through the pipeline: this file is about
    the endpoint, and a real ingestion run per test would make it slow and
    would retest what the integration suite already covers.
    """
    created_documents: list[uuid.UUID] = []
    created_users: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(content: str = PASSAGE, *, owner: User | None = None) -> Document:
        if owner is None:
            owner = db_session.execute(
                select(User).where(User.email == email)
            ).scalars().one()

        document = Document(
            id=uuid.uuid4(),
            user_id=owner.id,
            source=DocumentSource.ARTICLE,
            status=IngestionStatus.COMPLETED,
            content=content,
            title="Espresso notes",
            url="https://example.com/espresso",
        )
        db_session.add(document)
        db_session.flush()
        created_documents.append(document.id)

        db_session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content=content,
                start_char=0,
                end_char=len(content),
                embedding=[0.01] * EMBEDDING_DIM,
                embedding_model="fake",
            )
        )
        db_session.commit()
        return document

    def stranger() -> User:
        other = User(id=uuid.uuid4(), email=f"other-{uuid.uuid4().hex[:8]}@example.com")
        db_session.add(other)
        db_session.commit()
        created_users.append(other.id)
        return other

    build.stranger = stranger
    yield build

    for document_id in created_documents:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()
    for user_id in created_users:
        user = db_session.get(User, user_id)
        if user is not None:
            db_session.delete(user)
    db_session.commit()


def search(client, headers, **body):
    body.setdefault("query", "why does my espresso taste sour")
    return client.post("/api/v1/search", headers=headers, json=body)


def test_another_users_chunks_are_never_returned(
    client, auth_headers, make_chunk, fake_embedder
):
    """
    Chunks carry no owner of their own — the only place ownership can be
    enforced is the join to documents.
    """
    make_chunk("my own notes about espresso grind size and dose")
    make_chunk(
        "SOMEBODY ELSE'S PRIVATE RECEIPT, total 349.50 UAH",
        owner=make_chunk.stranger(),
    )

    body = search(client, auth_headers, top_k=50, min_similarity=-1.0).json()

    assert body["count"] >= 1, "the caller's own chunk should come back"
    for hit in body["hits"]:
        assert "SOMEBODY ELSE" not in hit["content"], "another user's text leaked"


def test_document_ids_cannot_widen_the_search(
    client, auth_headers, make_chunk, fake_embedder, db_session
):
    """
    Naming another user's document explicitly must not reach it. The
    filter narrows within your own documents; it is not a lookup key.
    """
    make_chunk("my own notes about espresso")
    theirs = make_chunk("SOMEBODY ELSE'S PRIVATE RECEIPT", owner=make_chunk.stranger())

    body = search(
        client,
        auth_headers,
        top_k=50,
        min_similarity=-1.0,
        document_ids=[str(theirs.id)],
    ).json()

    assert body["count"] == 0, "asking for another user's document returned something"


def test_authentication_is_required(client):
    response = client.post("/api/v1/search", json={"query": "anything"})
    assert response.status_code in (401, 403)


def test_a_hit_carries_everything_a_citation_needs(
    client, auth_headers, make_chunk, fake_embedder
):
    document = make_chunk()

    body = search(client, auth_headers, min_similarity=-1.0).json()

    assert body["count"] == 1
    hit = body["hits"][0]
    assert hit["document_id"] == str(document.id)
    assert hit["document_title"] == "Espresso notes"
    assert hit["document_url"] == "https://example.com/espresso"
    assert hit["chunk_index"] == 0
    assert -1.0 <= hit["similarity"] <= 1.0
    # The offsets index the DOCUMENT, which is what lets the citation
    # ticket map them onto a page without changing this response.
    assert document.content[hit["start_char"]: hit["end_char"]] == hit["content"]


def test_the_response_echoes_the_limit_that_was_applied(
    client, auth_headers, make_chunk, fake_embedder
):
    """
    top_k has a server-side default, so a client that sent nothing still
    needs to learn what limit it got
    """
    make_chunk()

    defaulted = search(client, auth_headers, min_similarity=-1.0).json()
    assert defaulted["top_k"] == 5, "the documented default changed"

    asked = search(client, auth_headers, top_k=3, min_similarity=-1.0).json()
    assert asked["top_k"] == 3
    assert asked["query"] == "why does my espresso taste sour"


def test_top_k_actually_caps_the_results(client, auth_headers, make_chunk, fake_embedder):
    for i in range(4):
        make_chunk(f"espresso note number {i} about grind size and dose")

    body = search(client, auth_headers, top_k=2, min_similarity=-1.0).json()

    assert len(body["hits"]) == 2, "top_k did not limit the rows that came back"
    assert body["count"] == 2


def test_no_matches_is_an_empty_list_not_an_error(client, auth_headers, fake_embedder):
    response = search(client, auth_headers, min_similarity=0.999999)
    assert response.status_code == 200
    assert response.json() == {
        "query": "why does my espresso taste sour",
        "top_k": 5,
        "count": 0,
        "hits": [],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", ""),
        ("query", "x" * 1001),
        ("top_k", 0),
        ("top_k", 101),
        ("min_similarity", -1.5),
        ("min_similarity", 1.5),
    ],
)
def test_bad_input_is_422_not_500(client, auth_headers, fake_embedder, field, value):
    """
    The service raises ValueError for an out-of-range top_k, which would
    surface as a 500. Validating at the edge turns the same mistake into a
    422 naming the field.
    """
    body = {"query": "a real question about espresso"}
    body[field] = value
    assert client.post("/api/v1/search", headers=auth_headers, json=body).status_code == 422


def test_an_unknown_field_is_refused(client, auth_headers, fake_embedder):
    """RequestSchema forbids extras, so a typo'd field fails loudly."""
    response = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "espresso", "topk": 3},
    )
    assert response.status_code == 422


def test_top_k_at_the_boundaries_is_accepted(client, auth_headers, make_chunk, fake_embedder):
    make_chunk()
    assert search(client, auth_headers, top_k=1).status_code == 200
    assert search(client, auth_headers, top_k=100).status_code == 200