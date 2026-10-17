import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Document, DocumentSource, IngestionStatus, User
from app.services.storage.fake_store import FakeFileStore
from app.services.storage.s3_store import get_file_store

PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 100, "needs_ocr": False, "source": "pdf"},
    {"index": 1, "start_char": 100, "end_char": 200, "needs_ocr": False, "source": "pdf"},
]


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
def stored_document(db_session, seeded_user, store):
    """A document whose bytes are in the (fake) store."""
    created: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(extension: str = "pdf", **fields) -> Document:
        user = db_session.execute(select(User).where(User.email == email)).scalars().one()
        document_id = uuid.uuid4()
        key = f"documents/{user.id}/{document_id}.{extension}"

        document = Document(
            id=document_id,
            user_id=user.id,
            source=fields.pop("source", DocumentSource.PDF),
            status=IngestionStatus.COMPLETED,
            content="x" * 200,
            title=fields.pop("title", "Invoice 2026-0042"),
            storage_key=key,
            file_size=1234,
            original_filename=fields.pop("original_filename", "invoice.pdf"),
            page_map=fields.pop("page_map", PAGE_MAP),
            **fields,
        )
        db_session.add(document)
        db_session.commit()
        created.append(document_id)

        store.objects[key] = b"%PDF-1.7 pretend"
        return document

    yield build

    for document_id in created:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()


def link(client, headers, document_id, **params):
    return client.get(f"/api/v1/documents/{document_id}/file", headers=headers, params=params)


def test_a_page_opens_the_pdf_at_that_page(client, auth_headers, stored_document):
    document = stored_document()
    body = link(client, auth_headers, document.id, page=2).json()

    assert body["url"].endswith("#page=2")


def test_the_fragment_goes_after_the_signed_query_string(client, auth_headers, stored_document):
    """
    A presigned URL already carries ?X-Amz-… parameters. The fragment has
    to come last to be a valid URL, and it cannot invalidate the
    signature because a fragment is never sent to the server at all.
    """
    document = stored_document()
    url = link(client, auth_headers, document.id, page=2).json()["url"]

    assert "?" in url
    assert url.index("?") < url.index("#")
    assert url.count("#") == 1


def test_without_a_page_there_is_no_fragment(client, auth_headers, stored_document):
    document = stored_document()
    assert "#" not in link(client, auth_headers, document.id).json()["url"]


def test_a_page_on_an_image_is_ignored_rather_than_refused(client, auth_headers, stored_document):
    """
    A client passes citation.pages[0] through without looking, and an
    image citation legitimately carries page 1. Refusing it would make
    every client special-case the source type before it could build a
    link.
    """
    document = stored_document(
        extension="png",
        source=DocumentSource.IMAGE,
        original_filename="receipt.png",
        page_map=[{"index": 0, "start_char": 0, "end_char": 200, "needs_ocr": False}],
    )
    body = link(client, auth_headers, document.id, page=1)

    assert body.status_code == 200
    assert "#" not in body.json()["url"]


def test_a_page_past_the_end_is_refused(client, auth_headers, stored_document):
    """
    #page=99 on a two-page PDF opens silently at page 2, so the user
    would read page 2 while the footnote said page 99 with no way to tell
    they were looking at the wrong thing.
    """
    document = stored_document()
    response = link(client, auth_headers, document.id, page=99)

    assert response.status_code == 422
    assert "2 page" in response.json()["error"]["message"]


def test_the_page_bound_is_exact(client, auth_headers, stored_document):
    """
    The boundary, not a number far past it.
    """
    document = stored_document()  # two pages

    assert link(client, auth_headers, document.id, page=2).status_code == 200
    assert link(client, auth_headers, document.id, page=3).status_code == 422


def test_a_page_is_allowed_when_the_page_count_is_unknown(client, auth_headers, stored_document):
    """
    Documents ingested before page_map existed have none. Refusing those
    would break verification for the older half of the library, so the
    check only runs when there is something to check against.
    """
    document = stored_document(page_map=None)
    body = link(client, auth_headers, document.id, page=7)

    assert body.status_code == 200
    assert body.json()["url"].endswith("#page=7")
    assert body.json()["page_count"] == 0


@pytest.mark.parametrize("page", [0, -1, "two"])
def test_a_nonsense_page_is_422(client, auth_headers, stored_document, page):
    document = stored_document()
    assert link(client, auth_headers, document.id, page=page).status_code == 422


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        ("pdf", "application/pdf"),
        ("png", "image/png"),
        ("jpg", "image/jpeg"),
        ("webp", "image/webp"),
        ("tiff", "image/tiff"),
    ],
)
def test_the_media_type_comes_back(client, auth_headers, stored_document, extension, expected):
    """
    A client cannot tell what it is being handed from a presigned URL
    whose path is a UUID, so it cannot decide between an <iframe>, an
    <img> and a download link. The type comes from the extension the
    upload path derived from the file's own magic bytes.
    """
    document = stored_document(extension=extension)
    assert link(client, auth_headers, document.id).json()["media_type"] == expected


def test_an_unrecognised_extension_is_not_claimed_to_be_a_pdf(
    client, auth_headers, stored_document
):
    """
    Falls back to octet-stream, which makes a client offer a download —
    the safe wrong answer. Claiming application/pdf for something else
    would be the unsafe one.
    """
    document = stored_document(extension="bin")
    assert link(client, auth_headers, document.id).json()["media_type"] == (
        "application/octet-stream"
    )


def test_the_filename_and_page_count_come_back(client, auth_headers, stored_document):
    document = stored_document()
    body = link(client, auth_headers, document.id).json()

    assert body["filename"] == "invoice.pdf"
    assert body["page_count"] == 2
    assert body["expires_in"] == 900


def test_a_captured_document_has_no_filename(client, auth_headers, stored_document):
    """Articles and selections were never uploaded, so there is no name
    the user would recognise. None, not an invented one."""
    document = stored_document(original_filename=None)
    assert link(client, auth_headers, document.id).json()["filename"] is None
