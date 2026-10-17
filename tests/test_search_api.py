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
            owner = db_session.execute(select(User).where(User.email == email)).scalars().one()

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


def test_another_users_chunks_are_never_returned(client, auth_headers, make_chunk, fake_embedder):
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


def test_a_hit_carries_everything_a_citation_needs(client, auth_headers, make_chunk, fake_embedder):
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
    assert document.content[hit["start_char"] : hit["end_char"]] == hit["content"]


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
        "retrieval": {
            "citations": [],
            "documents": [],
            "tags": [],
            "document_count": 0,
            "chunk_count": 0,
        },
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


@pytest.fixture
def make_multi_chunk(db_session, seeded_user):
    """
    make_chunk gives one chunk at offset zero, which cannot show whether
    the grouping works — and grouping is the whole point of the
    `documents` layer.
    """
    created: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(*, page_map=None, title="Invoice 2026-0042", source=DocumentSource.PDF):
        user = db_session.execute(select(User).where(User.email == email)).scalars().one()
        content = ("Espresso grind and dose notes. " * 20)[:600]

        document = Document(
            id=uuid.uuid4(),
            user_id=user.id,
            source=source,
            status=IngestionStatus.COMPLETED,
            content=content,
            title=title,
            page_map=page_map,
        )
        db_session.add(document)
        db_session.flush()
        created.append(document.id)

        for index, (start, end) in enumerate([(0, 300), (300, 600)]):
            db_session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=content[start:end],
                    start_char=start,
                    end_char=end,
                    embedding=[0.01] * EMBEDDING_DIM,
                    embedding_model="fake",
                )
            )
        db_session.commit()
        return document

    yield build

    for document_id in created:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()


def test_search_returns_the_retrieval_object_the_frontend_renders(
    client, auth_headers, make_chunk, fake_embedder
):
    """
    /search used to return SearchHitRead only — no locator, no pages, no
    from_ocr, no tags. The frontend contract told the engineer to build
    the sources panel against this endpoint, so he would have built
    against one shape and rebuilt for another when /ask landed.
    """
    make_chunk()
    retrieval = search(client, auth_headers, min_similarity=-1.0).json()["retrieval"]

    assert retrieval["chunk_count"] == 1
    citation = retrieval["citations"][0]
    # Every field the contract promises, present rather than merely not null.
    for field in (
        "marker",
        "locator",
        "pages",
        "page_count",
        "from_ocr",
        "ocr_confidence",
        "start_char",
        "end_char",
        "quote",
        "similarity",
        "url",
    ):
        assert field in citation, f"the frontend contract promises {field}"


def test_markers_are_one_to_n_with_no_gaps(client, auth_headers, make_multi_chunk, fake_embedder):
    """A gap would look to a reader like a footnote that went missing."""
    make_multi_chunk()
    body = search(client, auth_headers, min_similarity=-1.0).json()

    markers = [c["marker"] for c in body["retrieval"]["citations"]]
    assert markers == [f"[{i}]" for i in range(1, len(markers) + 1)]
    assert len(markers) == body["count"]


def test_two_chunks_from_one_document_are_one_chip(
    client, auth_headers, make_multi_chunk, fake_embedder
):
    """
    Rendering one chip per citation would show the same invoice twice.
    This grouping is what prevents it, and it has to survive the trip
    through the API, not just the service layer.
    """
    make_multi_chunk(
        page_map=[
            {"index": 0, "start_char": 0, "end_char": 300, "needs_ocr": False, "source": "pdf"},
            {
                "index": 1,
                "start_char": 300,
                "end_char": 600,
                "needs_ocr": False,
                "source": "ocr",
                "mean_confidence": 55.0,
            },
        ]
    )
    retrieval = search(client, auth_headers, min_similarity=-1.0).json()["retrieval"]

    assert retrieval["chunk_count"] == 2
    assert retrieval["document_count"] == 1

    chip = retrieval["documents"][0]
    assert chip["markers"] == ["[1]", "[2]"]
    assert chip["pages"] == [1, 2]
    assert chip["locator"] == "pages 1-2"
    # One cited passage was machine-read, so the chip carries the warning.
    assert chip["from_ocr"] is True


def test_hits_and_citations_describe_the_same_chunks(
    client, auth_headers, make_multi_chunk, fake_embedder
):
    """
    Both are returned, so they must agree. If they ever disagree the
    frontend has no way to know which one lied.
    """
    make_multi_chunk()
    body = search(client, auth_headers, min_similarity=-1.0).json()

    assert [h["chunk_id"] for h in body["hits"]] == [
        c["chunk_id"] for c in body["retrieval"]["citations"]
    ]
    for hit, citation in zip(body["hits"], body["retrieval"]["citations"], strict=True):
        assert (hit["start_char"], hit["end_char"]) == (
            citation["start_char"],
            citation["end_char"],
        )
        assert hit["similarity"] == citation["similarity"]


def test_the_retrieval_object_is_scoped_to_the_caller_too(
    client, auth_headers, make_chunk, fake_embedder
):
    """
    The isolation guarantee has to cover the new block as well. It is
    built from the hits, so it should — but "should" is how a leak gets
    shipped, and this block now carries document titles and tag names
    rather than only chunk text.
    """
    make_chunk("my own notes about espresso grind size and dose")
    make_chunk(
        "SOMEBODY ELSE'S PRIVATE RECEIPT, total 349.50 UAH",
        owner=make_chunk.stranger(),
    )

    retrieval = search(client, auth_headers, top_k=50, min_similarity=-1.0).json()["retrieval"]

    for citation in retrieval["citations"]:
        assert "PRIVATE RECEIPT" not in citation["quote"]
    assert retrieval["document_count"] == 1
