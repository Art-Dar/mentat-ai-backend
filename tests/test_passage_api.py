import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import get_embedder
from app.services.ingestion.fake_embedder import FakeEmbedder

ARTICLE = (
    "Espresso extraction depends on grind size, dose and water temperature. "
    "A shot pulled too fast tastes sour because the water has not had time "
    "to dissolve the sugars locked inside the puck. Grinding finer raises "
    "resistance and slows the flow, which is the usual first correction. "
    "Temperature matters less than people assume, within a degree or two."
)

PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 140, "needs_ocr": False, "source": "pdf"},
    {
        "index": 1,
        "start_char": 140,
        "end_char": len(ARTICLE),
        "needs_ocr": False,
        "source": "ocr",
        "mean_confidence": 57.0,
    },
]


@pytest.fixture
def fake_embedder():
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
def make_document(db_session, seeded_user):
    created_documents: list[uuid.UUID] = []
    created_users: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(
        content: str | None = ARTICLE,
        *,
        owner: User | None = None,
        with_chunk: bool = False,
        **fields,
    ) -> Document:
        if owner is None:
            owner = db_session.execute(select(User).where(User.email == email)).scalars().one()

        document = Document(
            id=uuid.uuid4(),
            user_id=owner.id,
            source=fields.pop("source", DocumentSource.ARTICLE),
            status=fields.pop("status", IngestionStatus.COMPLETED),
            content=content,
            title=fields.pop("title", "Espresso notes"),
            **fields,
        )
        db_session.add(document)
        db_session.flush()
        created_documents.append(document.id)

        if with_chunk:
            # A chunk in the middle of the document, so start_char is a
            # real offset rather than zero — zero would pass even if the
            # endpoint ignored the parameter entirely.
            start = ARTICLE.index("Grinding finer")
            db_session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=0,
                    content=ARTICLE[start : start + 120],
                    start_char=start,
                    end_char=start + 120,
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


def passage(client, headers, document_id, **params):
    return client.get(f"/api/v1/documents/{document_id}/passage", headers=headers, params=params)


def test_a_citation_from_search_lands_on_the_text_it_quoted(
    client, auth_headers, make_document, fake_embedder
):
    """
    Search produces a citation; the citation's own start_char/end_char go
    straight back in; the highlight that comes out is the passage the
    citation quoted
    """
    document = make_document(page_map=PAGE_MAP, with_chunk=True)

    body = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "why is my shot sour", "min_similarity": -1.0},
    ).json()
    citation = body["retrieval"]["citations"][0]

    found = passage(
        client,
        auth_headers,
        document.id,
        start=citation["start_char"],
        end=citation["end_char"],
    )
    assert found.status_code == 200
    found = found.json()

    highlight = found["text"][found["highlight_start"] : found["highlight_end"]]
    assert highlight == ARTICLE[citation["start_char"] : citation["end_char"]]

    # The quote is the truncated, whitespace-collapsed form of the same
    # passage, so its opening words must appear in the highlight.
    assert citation["quote"].split("…")[0][:40] in " ".join(highlight.split())

    # And the pages agree, which is what shared span_provenance buys.
    assert found["pages"] == citation["pages"]
    assert found["from_ocr"] == citation["from_ocr"]
    assert found["ocr_confidence"] == citation["ocr_confidence"]


def test_the_window_widens_on_request(client, auth_headers, make_document):
    document = make_document()
    start = ARTICLE.index("Grinding finer")

    narrow = passage(client, auth_headers, document.id, start=start, end=start + 20, window=0)
    wide = passage(client, auth_headers, document.id, start=start, end=start + 20, window=300)

    assert len(wide.json()["text"]) > len(narrow.json()["text"])
    assert narrow.json()["text"] == ARTICLE[start : start + 20]

    # Same span either way; only the surrounding context grew. If the
    # window shifted the highlight, a user widening the context would
    # watch the quoted sentence change underneath them.
    widened = wide.json()
    assert (
        widened["text"][widened["highlight_start"] : widened["highlight_end"]]
        == narrow.json()["text"]
    )


def test_chars_before_and_after_tell_the_client_whether_to_offer_more(
    client, auth_headers, make_document
):
    document = make_document()
    body = passage(client, auth_headers, document.id, start=200, end=240, window=10).json()

    assert body["chars_before"] > 0
    assert body["chars_after"] > 0
    assert body["char_count"] == len(ARTICLE)


def test_offsets_past_the_end_are_409_not_500(client, auth_headers, make_document):
    """
    The document was re-ingested after the citation was built. The client
    did nothing wrong, so this is a conflict rather than a bad request —
    and the distinction matters because the recovery is different: re-run
    the search, do not fix your parameters.
    """
    document = make_document()
    response = passage(client, auth_headers, document.id, start=99_000, end=99_100)

    assert response.status_code == 409
    assert "re-processed" in response.json()["error"]["message"]


def test_a_document_still_being_processed_is_409(client, auth_headers, make_document):
    """No extracted text yet, so there is nothing to show. The user gets
    told, rather than an empty pane that looks like a bug."""
    document = make_document(content=None, status=IngestionStatus.PENDING)
    response = passage(client, auth_headers, document.id, start=0, end=50)

    assert response.status_code == 409
    assert "no extracted text" in response.json()["error"]["message"].lower()


def test_another_users_document_is_404_not_403(client, auth_headers, make_document):
    """
    Same reasoning as the rest of the document routes: a 403 would confirm
    the id is real. This endpoint returns document TEXT, so getting it
    wrong leaks the contents rather than merely the existence.
    """
    document = make_document(owner=make_document.stranger())
    response = passage(client, auth_headers, document.id, start=0, end=50)

    assert response.status_code == 404


def test_a_missing_document_is_404(client, auth_headers):
    assert passage(client, auth_headers, uuid.uuid4(), start=0, end=50).status_code == 404


def test_authentication_is_required(client, make_document):
    document = make_document()
    response = client.get(
        f"/api/v1/documents/{document.id}/passage", params={"start": 0, "end": 50}
    )
    assert response.status_code in (401, 403)


@pytest.mark.parametrize(
    "params",
    [
        {},  # no span at all
        {"start": 0},  # half a span
        {"end": 50},
        {"start": -1, "end": 50},
        {"start": 0, "end": 0},
        {"start": 0, "end": 50, "window": -1},
        {"start": 0, "end": 50, "window": 99_999},
        {"start": "beginning", "end": 50},
    ],
)
def test_a_malformed_span_is_422(client, auth_headers, make_document, params):
    document = make_document()
    assert passage(client, auth_headers, document.id, **params).status_code == 422
