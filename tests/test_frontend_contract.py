import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import (
    Chunk,
    Document,
    DocumentSource,
    DocumentTag,
    IngestionStatus,
    Tag,
    User,
)
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import get_embedder
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.storage.fake_store import FakeFileStore
from app.services.storage.s3_store import get_file_store

# --- interface Citation
CITATION_KEYS = {
    "marker",
    "document_id",
    "chunk_id",
    "title",
    "source",
    "locator",
    "pages",
    "page_count",
    "start_char",
    "end_char",
    "from_ocr",
    "ocr_confidence",
    "similarity",
    "url",
    "quote",
}

# --- interface DocumentRef
DOCUMENT_REF_KEYS = {
    "document_id",
    "title",
    "source",
    "markers",
    "pages",
    "page_count",
    "locator",
    "has_file",
    "url",
    "from_ocr",
    "tags",
}

# --- interface Retrieval
RETRIEVAL_KEYS = {"citations", "documents", "tags", "document_count", "chunk_count"}

# --- interface TagRef
TAG_REF_KEYS = {"id", "name", "color"}

# --- interface PassageResponse
PASSAGE_KEYS = {
    "document_id",
    "title",
    "source",
    "text",
    "highlight_start",
    "highlight_end",
    "start_char",
    "end_char",
    "chars_before",
    "chars_after",
    "char_count",
    "pages",
    "page_count",
    "from_ocr",
    "ocr_confidence",
}

# --- the file link response
FILE_LINK_KEYS = {"url", "expires_in", "media_type", "filename", "page_count"}

# --- POST /search
SEARCH_KEYS = {"query", "top_k", "count", "hits", "retrieval"}

CONTENT = "Espresso grind and dose notes for the invoice. " * 14

# Page 1 is a text layer, page 2 was OCRed at 61% — so one document
# exercises the clean path, the OCR badge and the page range at once.
PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 320, "needs_ocr": False, "source": "pdf"},
    {
        "index": 1,
        "start_char": 320,
        "end_char": len(CONTENT),
        "needs_ocr": False,
        "source": "ocr",
        "mean_confidence": 61.0,
    },
]


@pytest.fixture
def offline():
    """No Voyage, no R2."""
    store = FakeFileStore()
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_file_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_embedder, None)
    app.dependency_overrides.pop(get_file_store, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def library(db_session, seeded_user, offline):
    """A tagged two-page PDF with two chunks and its bytes in the store."""
    email, _ = seeded_user
    user = db_session.execute(select(User).where(User.email == email)).scalars().one()

    document_id = uuid.uuid4()
    key = f"documents/{user.id}/{document_id}.pdf"
    offline.objects[key] = b"%PDF-1.7 pretend"

    document = Document(
        id=document_id,
        user_id=user.id,
        source=DocumentSource.PDF,
        status=IngestionStatus.COMPLETED,
        content=CONTENT,
        title="Invoice 2026-0042",
        storage_key=key,
        file_size=16,
        original_filename="invoice.pdf",
        url="https://example.com/invoice",
        page_map=PAGE_MAP,
    )
    db_session.add(document)
    db_session.flush()

    for index, (start, end) in enumerate([(0, 300), (300, 620)]):
        db_session.add(
            Chunk(
                document_id=document.id,
                chunk_index=index,
                content=CONTENT[start:end],
                start_char=start,
                end_char=end,
                embedding=[0.01] * EMBEDDING_DIM,
                embedding_model="fake",
            )
        )

    tag = Tag(id=uuid.uuid4(), name=f"Finance-{uuid.uuid4().hex[:4]}")
    db_session.add(tag)
    db_session.flush()
    db_session.add(DocumentTag(document_id=document.id, tag_id=tag.id))
    db_session.commit()

    yield document

    db_session.delete(db_session.get(Document, document_id))
    db_session.commit()
    db_session.delete(db_session.get(Tag, tag.id))
    db_session.commit()


@pytest.fixture
def retrieval(client, auth_headers, library):
    body = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "what is the total", "min_similarity": -1.0},
    ).json()
    assert set(body) == SEARCH_KEYS, set(body) ^ SEARCH_KEYS
    return body["retrieval"]


def test_the_retrieval_object_has_exactly_the_documented_keys(retrieval):
    assert set(retrieval) == RETRIEVAL_KEYS, set(retrieval) ^ RETRIEVAL_KEYS


def test_a_citation_has_exactly_the_documented_keys(retrieval):
    actual = set(retrieval["citations"][0])
    assert actual == CITATION_KEYS, actual ^ CITATION_KEYS


def test_a_document_reference_has_exactly_the_documented_keys(retrieval):
    actual = set(retrieval["documents"][0])
    assert actual == DOCUMENT_REF_KEYS, actual ^ DOCUMENT_REF_KEYS


def test_a_tag_reference_has_exactly_the_documented_keys(retrieval):
    assert set(retrieval["tags"][0]) == TAG_REF_KEYS
    # The same shape nested inside a document, so a client can use one
    # component for both.
    assert set(retrieval["documents"][0]["tags"][0]) == TAG_REF_KEYS


def test_a_passage_has_exactly_the_documented_keys(client, auth_headers, library, retrieval):
    citation = retrieval["citations"][0]
    body = client.get(
        f"/api/v1/documents/{library.id}/passage",
        headers=auth_headers,
        params={"start": citation["start_char"], "end": citation["end_char"]},
    ).json()

    assert set(body) == PASSAGE_KEYS, set(body) ^ PASSAGE_KEYS


def test_a_file_link_has_exactly_the_documented_keys(client, auth_headers, library):
    body = client.get(f"/api/v1/documents/{library.id}/file", headers=auth_headers).json()

    assert set(body) == FILE_LINK_KEYS, set(body) ^ FILE_LINK_KEYS


def test_passing_a_citations_pages_through_opens_the_pdf_there(
    client, auth_headers, library, retrieval
):
    """The document says to pass citation.pages[0] straight to ?page=."""
    citation = retrieval["citations"][0]
    body = client.get(
        f"/api/v1/documents/{library.id}/file",
        headers=auth_headers,
        params={"page": citation["pages"][0]},
    ).json()

    assert body["url"].endswith(f"#page={citation['pages'][0]}")
    assert body["media_type"] == "application/pdf"
    assert body["expires_in"] == 900, "the document promises a 15-minute link"


def test_slicing_the_passage_by_its_offsets_yields_the_cited_text(
    client, auth_headers, library, retrieval
):
    """
    "Render text and wrap text.slice(highlight_start, highlight_end)".
    If that instruction is wrong, every evidence pane shows the wrong
    sentence and looks entirely normal doing it.
    """
    citation = retrieval["citations"][0]
    body = client.get(
        f"/api/v1/documents/{library.id}/passage",
        headers=auth_headers,
        params={"start": citation["start_char"], "end": citation["end_char"]},
    ).json()

    assert (
        body["text"][body["highlight_start"] : body["highlight_end"]]
        == CONTENT[citation["start_char"] : citation["end_char"]]
    )
    assert body["pages"] == citation["pages"], "the two must not disagree"
    assert body["from_ocr"] == citation["from_ocr"]


def test_markers_join_the_prose_to_the_chips(retrieval):
    """
    "hover a chip → highlight every marker in its markers array". Both
    chunks come from one document, so its chip must carry both markers.
    """
    assert retrieval["document_count"] == 1
    assert retrieval["documents"][0]["markers"] == [c["marker"] for c in retrieval["citations"]]
    assert retrieval["documents"][0]["locator"] == "pages 1-2"
    assert retrieval["documents"][0]["from_ocr"] is True


def test_a_web_source_links_into_the_page_not_at_it(retrieval):
    """The `#:~:text=` claim."""
    assert "#:~:text=" in retrieval["citations"][0]["url"]


def test_nothing_found_is_empty_arrays_rather_than_nulls(client, auth_headers, offline):
    """
    "Every key in retrieval is always present. Nothing is null where an
    empty array would do, so you never have to branch on both."
    """
    body = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "nothing in the library matches this", "min_similarity": 0.9999},
    ).json()

    assert body["retrieval"] == {
        "citations": [],
        "documents": [],
        "tags": [],
        "document_count": 0,
        "chunk_count": 0,
    }
