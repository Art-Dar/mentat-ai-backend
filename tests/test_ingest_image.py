import io
import shutil
import uuid

import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select

from app.main import app
from app.models import Document, DocumentSource, IngestionStatus
from app.services.ingestion.extractors.ocr import (
    OcrEngineUnavailable,
    OcrExtraction,
    OcrWord,
    UnreadableImageError,
)
from app.services.storage.base import storage_key
from app.services.storage.fake_store import BrokenFileStore, FakeFileStore
from app.services.storage.s3_store import get_file_store

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="the tesseract binary is not installed on this host",
)


def message_of(response) -> str:
    """
    The human-readable part of this API's error envelope.

    register_error_handling rewrites every error into
    {"error": {"code", "message", "details", "request_id"}}, so a
    FastAPI-style response.json()["detail"] does not exist here.
    """
    return response.json()["error"]["message"]


RECEIPT = ["ATB Market", "Chek 10482", "Razom: 349.50 UAH", "Data: 16.10.2026"]


def _font(size: int = 34):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    pytest.skip("no scalable font available to render a test scan")


def photo(lines: list[str] = RECEIPT, *, fmt: str = "PNG", size=(1000, 500)) -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 52
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def store():
    fake = FakeFileStore()
    app.dependency_overrides[get_file_store] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_file_store, None)


@pytest.fixture
def broken_store():
    app.dependency_overrides[get_file_store] = lambda: BrokenFileStore()
    yield
    app.dependency_overrides.pop(get_file_store, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def upload(client, auth_headers, db_session, store):
    """
    Depends on `store` for the same reason the PDF fixture does: without
    the override a real S3FileStore is built and the test writes into the
    live bucket.
    """
    created: list[uuid.UUID] = []

    # Named `content`, not `data`: httpx uses `data=` for form fields, and
    # the language tests need to pass one through **extra.
    def post(content: bytes | None = None, filename: str = "receipt.png", **extra):
        response = client.post(
            "/api/v1/ingest/image",
            headers=auth_headers,
            files={"file": (filename, content if content is not None else photo(), "image/png")},
            **extra,
        )
        if response.status_code == 201:
            created.append(uuid.UUID(response.json()["document_id"]))
        return response

    yield post

    for document_id in created:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()


@needs_tesseract
def test_a_receipt_photo_is_ingested(upload, db_session, store):
    response = upload()
    assert response.status_code == 201, response.text

    body = response.json()
    assert body["status"] == IngestionStatus.PENDING
    assert body["char_count"] >= 16
    assert body["word_count"] >= 6
    assert body["mean_confidence"] > 60
    assert body["low_confidence"] is False
    assert body["language"] == "eng"

    document = db_session.get(Document, uuid.UUID(body["document_id"]))
    assert document.source == DocumentSource.IMAGE
    assert "349.50" in document.content
    assert document.original_filename == "receipt.png"
    assert document.file_size == len(photo())


@needs_tesseract
def test_the_original_bytes_are_stored_under_the_document_id(upload, db_session, store):
    data = photo()
    body = upload(data).json()
    document = db_session.get(Document, uuid.UUID(body["document_id"]))

    expected = storage_key(document.user_id, document.id, extension="png")
    assert document.storage_key == expected
    assert store.objects[expected] == data, "the stored object must be the uploaded bytes"


@needs_tesseract
def test_the_page_map_describes_the_ocr_pass(upload, db_session):
    """
    page_map means the same thing for both sources: which span of content
    came from where. needs_ocr is False because this text IS the OCR
    output — there is nothing left to queue.
    """
    body = upload().json()
    document = db_session.get(Document, uuid.UUID(body["document_id"]))

    assert len(document.page_map) == 1
    entry = document.page_map[0]
    assert entry["index"] == 0
    assert entry["source"] == "ocr"
    assert entry["needs_ocr"] is False
    assert entry["start_char"] == 0
    assert entry["end_char"] == len(document.content)
    assert document.content[entry["start_char"]: entry["end_char"]] == document.content
    assert entry["language"] == "eng"
    assert entry["word_count"] == body["word_count"]


@needs_tesseract
@pytest.mark.parametrize(
    ("fmt", "extension"),
    [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp"), ("TIFF", "tiff")],
)
def test_the_extension_comes_from_the_magic_bytes(upload, db_session, fmt, extension):
    """
    Not from the filename. A file called receipt.png that is really a JPEG
    must be stored as .jpg, or the presigned URL serves bytes that do not
    match the key and browsers guess wrong.
    """
    body = upload(photo(fmt=fmt), filename="receipt.png").json()
    document = db_session.get(Document, uuid.UUID(body["document_id"]))
    assert document.storage_key.endswith(f".{extension}")


@needs_tesseract
@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),  # not image/jpg, which is not a real media type
        ("WEBP", "image/webp"),
        ("TIFF", "image/tiff"),
    ],
)
def test_the_stored_media_type_comes_from_the_bytes(upload, db_session, store, fmt, media_type):
    body = upload(photo(fmt=fmt)).json()
    document = db_session.get(Document, uuid.UUID(body["document_id"]))
    assert store.content_types[document.storage_key] == media_type


@needs_tesseract
def test_a_lying_client_content_type_is_ignored(client, auth_headers, db_session, store):
    """
    The stored media type is served back verbatim on the presigned URL, so
    taking it from the request would let a client upload a PNG while
    claiming text/html and have that HTML served from a URL this app hands
    out. Stored XSS, delivered by us. The bytes decide.
    """
    response = client.post(
        "/api/v1/ingest/image",
        headers=auth_headers,
        files={"file": ("receipt.png", photo(), "text/html")},
    )
    assert response.status_code == 201

    document_id = uuid.UUID(response.json()["document_id"])
    document = db_session.get(Document, document_id)
    assert store.content_types[document.storage_key] == "image/png"

    db_session.delete(document)
    db_session.commit()


@needs_tesseract
def test_the_stored_image_is_retrievable_through_the_file_endpoint(
        upload, client, auth_headers
):
    """The GET endpoint is shared with the PDF path and needs no changes."""
    document_id = upload().json()["document_id"]

    response = client.get(f"/api/v1/documents/{document_id}/file", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["url"].startswith("https://fake-store.invalid/")


def test_a_pdf_is_refused_by_the_image_endpoint(upload):
    response = upload(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", filename="invoice.pdf")
    assert response.status_code == 415
    assert "PNG" in message_of(response)


def test_a_text_file_renamed_to_png_is_refused(upload):
    """The filename is the client's claim; the first bytes are the file."""
    response = upload(b"this is not an image", filename="receipt.png")
    assert response.status_code == 415


def test_an_oversized_upload_is_refused(upload):
    response = upload(b"\x89PNG\r\n\x1a\n" + b"\x00" * (16 * 1024 * 1024))
    assert response.status_code == 413


def test_authentication_is_required(client):
    response = client.post(
        "/api/v1/ingest/image",
        files={"file": ("receipt.png", photo(), "image/png")},
    )
    assert response.status_code in (401, 403)


@pytest.mark.parametrize(
    "language", ["english", "EN", "../../etc/passwd", "eng;rm -rf /", "eng+"]
)
def test_a_malformed_language_code_is_refused(upload, language):
    response = upload(data={"language": language})
    assert response.status_code == 422
    assert "tesseract code" in message_of(response)


@needs_tesseract
@pytest.mark.parametrize("language", ["eng", "eng+eng"])
def test_a_well_formed_language_code_is_accepted(upload, language):
    assert upload(data={"language": language}).status_code == 201


@needs_tesseract
def test_an_empty_language_field_falls_back_to_the_default(upload):
    """
    Pinning behaviour rather than asserting a preference: an empty
    multipart value does not reach the handler as "", so Form's default
    applies and the request succeeds in English. Worth a test because the
    obvious guess — that "" fails the regex and 422s — is wrong.
    """
    response = upload(data={"language": ""})
    assert response.status_code == 201
    assert response.json()["language"] == "eng"


def _fake_extraction(text: str, confidence: float = 90.0) -> OcrExtraction:
    words = tuple(
        OcrWord(word, confidence, 0, 0, 10, 10) for word in text.split() if word
    )
    return OcrExtraction(text=text, words=words, language="eng")


def test_an_image_with_no_readable_text_is_refused(upload, monkeypatch, store):
    """
    422 before anything is written. A document with empty content would be
    invisible in search and indistinguishable from a failed upload, so the
    user is told while they still have the receipt in their hand.
    """

    async def blank(data, *, language="eng"):
        return _fake_extraction("")

    monkeypatch.setattr("app.api.v1.files.extract_image_text_async", blank)

    response = upload()
    assert response.status_code == 422
    assert "No readable text" in message_of(response)
    assert store.objects == {}, "nothing may be stored for a rejected image"


def test_low_confidence_text_is_accepted_but_flagged(upload, monkeypatch, db_session):
    """
    Kept, not refused: barely-readable text is still worth more than
    nothing, and the flag lets the client offer a retake.
    """

    async def poor(data, *, language="eng"):
        return _fake_extraction("Razom 349.50 UAH ATB Market", confidence=41.0)

    monkeypatch.setattr("app.api.v1.files.extract_image_text_async", poor)

    body = upload().json()
    assert body["low_confidence"] is True
    assert body["mean_confidence"] == pytest.approx(41.0)
    assert db_session.get(Document, uuid.UUID(body["document_id"])) is not None


def test_a_missing_tesseract_binary_is_a_503_not_a_422(upload, monkeypatch):
    """
    The upload was fine and the user has nothing to fix, so this must not
    be reported as a bad request. It is the distinction _engine_failure
    exists to draw.
    """

    async def unavailable(data, *, language="eng"):
        raise OcrEngineUnavailable("the tesseract binary is not installed on this host")

    monkeypatch.setattr("app.api.v1.files.extract_image_text_async", unavailable)

    response = upload()
    assert response.status_code == 503
    assert "unavailable" in message_of(response).lower()


def test_a_corrupt_image_is_a_422(upload, monkeypatch):
    async def corrupt(data, *, language="eng"):
        raise UnreadableImageError("image data is incomplete")

    monkeypatch.setattr("app.api.v1.files.extract_image_text_async", corrupt)

    response = upload()
    assert response.status_code == 422


def test_storage_failure_leaves_no_row_behind(upload, broken_store, monkeypatch, db_session):
    """
    Bytes are written before the row, so a storage failure must abort
    before anything is committed — the reverse order would leave a visible
    document whose file 404s.
    """

    async def fine(data, *, language="eng"):
        return _fake_extraction("Razom 349.50 UAH ATB Market Chek")

    monkeypatch.setattr("app.api.v1.files.extract_image_text_async", fine)

    before = db_session.execute(
        select(Document).where(Document.source == DocumentSource.IMAGE)
    ).scalars().all()

    response = upload()
    assert response.status_code == 503

    after = db_session.execute(
        select(Document).where(Document.source == DocumentSource.IMAGE)
    ).scalars().all()
    assert len(after) == len(before)