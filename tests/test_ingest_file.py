import io
import uuid

import pytest
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlalchemy import select

from app.main import app
from app.models import Document, DocumentSource, IngestionStatus
from app.services.storage.fake_store import BrokenFileStore, FakeFileStore
from app.services.storage.s3_store import get_file_store

INVOICE = [
    "INVOICE 2026-0042",
    "Vendor: Kyiv Coffee Roasters",
    "Total: 480.00 UAH",
    "Payment terms: net 14 days from the invoice date",
]


def text_pdf(*pages: list[str]) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(60, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def scan_pdf() -> bytes:
    image = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(image).text((80, 80), "RECEIPT 77/12 ATB 212.45 UAH", fill="black")
    png = io.BytesIO()
    image.save(png, "PNG")
    png.seek(0)
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.drawImage(ImageReader(png), 0, 0, width=A4[0], height=A4[1])
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


@pytest.fixture
def store():
    """A FakeFileStore wired in for the duration of one test."""
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
    POST a file and clean the document up afterwards.

    Depends on `store` so the FakeFileStore override is ALWAYS installed.
    Without it, a test that uses `upload` alone builds a real S3FileStore
    and talks to R2 — passing on a laptop with credentials, failing in CI,
    and quietly writing test PDFs into the real bucket either way.
    """
    created: list[uuid.UUID] = []

    def post(data: bytes = None, filename: str = "invoice.pdf", **extra):
        response = client.post(
            "/api/v1/ingest/file",
            headers=auth_headers,
            files={
                "file": (
                    filename,
                    data if data is not None else text_pdf(INVOICE),
                    "application/pdf",
                )
            },
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


# the happy path


def test_a_pdf_is_accepted(upload):
    response = upload()

    assert response.status_code == 201
    assert response.json()["status"] == "pending"


def test_the_document_row_carries_the_extracted_text(upload, db_session):
    response = upload()
    document = db_session.get(Document, uuid.UUID(response.json()["document_id"]))

    assert document.source is DocumentSource.PDF
    assert "INVOICE 2026-0042" in document.content
    assert document.status is IngestionStatus.PENDING


def test_the_bytes_reach_the_store(upload, store):
    data = text_pdf(INVOICE)
    response = upload(data)
    document_id = response.json()["document_id"]

    assert len(store.objects) == 1
    key, stored = next(iter(store.objects.items()))
    assert stored == data
    assert document_id in key
    assert store.content_types[key] == "application/pdf"


def test_the_storage_key_is_recorded_on_the_row(upload, db_session, store):
    response = upload()
    document = db_session.get(Document, uuid.UUID(response.json()["document_id"]))

    assert document.storage_key in store.objects
    assert document.file_size > 0


def test_the_key_is_built_from_ids_not_the_filename(upload, store):
    """Path traversal: a hostile filename must never reach the key."""
    upload(filename="../../etc/passwd.pdf")

    key = next(iter(store.objects))
    assert ".." not in key
    assert "passwd" not in key


def test_the_original_filename_is_kept_for_display(upload, db_session):
    response = upload(filename="march-invoice.pdf")
    document = db_session.get(Document, uuid.UUID(response.json()["document_id"]))

    assert document.original_filename == "march-invoice.pdf"


def test_the_page_map_is_stored_for_citations(upload, db_session):
    response = upload(text_pdf(INVOICE, INVOICE))
    document = db_session.get(Document, uuid.UUID(response.json()["document_id"]))

    assert len(document.page_map) == 2
    for page in document.page_map:
        assert document.content[page["start_char"] : page["end_char"]] is not None
        assert set(page) == {"index", "start_char", "end_char", "needs_ocr"}


def test_page_offsets_still_index_the_stored_content(upload, db_session):
    """The guarantee a citation relies on, preserved through the endpoint."""
    response = upload(text_pdf(INVOICE, INVOICE))
    document = db_session.get(Document, uuid.UUID(response.json()["document_id"]))

    for page in document.page_map:
        assert document.content[page["start_char"] : page["end_char"]].strip()


# the OCR verdict reaches the client


def test_a_scan_is_reported_as_needing_ocr(upload):
    response = upload(scan_pdf())

    body = response.json()
    assert response.status_code == 201
    assert body["needs_ocr"] is True
    assert body["pages_needing_ocr"] == [0]


def test_a_text_pdf_does_not_need_ocr(upload):
    body = upload().json()

    assert body["needs_ocr"] is False
    assert body["pages_needing_ocr"] == []
    assert body["page_count"] == 1


# refusals


def test_a_non_pdf_is_415(upload):
    response = upload(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200, filename="photo.png")

    assert response.status_code == 415


def test_a_renamed_non_pdf_is_still_415(upload):
    """Content-Type and extension are both client-supplied. Magic bytes are not."""
    response = upload(b"not a pdf at all", filename="invoice.pdf")

    assert response.status_code == 415


def test_a_corrupt_pdf_is_422(upload):
    response = upload(b"%PDF-1.7\nthis is not a pdf at all\n")

    assert response.status_code == 422


def test_an_oversized_file_is_413(upload):
    response = upload(b"%PDF-" + b"\x00" * (26 * 1024 * 1024))

    assert response.status_code == 413


def test_nothing_is_stored_when_the_file_is_refused(upload, store):
    upload(b"\x89PNG\r\n\x1a\n")

    assert store.objects == {}


def test_the_endpoint_requires_authentication(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("x.pdf", text_pdf(INVOICE), "application/pdf")},
    )

    assert response.status_code in (401, 403)


# the ordering that matters


def test_no_row_is_written_when_the_store_is_down(client, auth_headers, db_session, broken_store):
    """
    Bytes first, then the row. Proven by breaking the store: if the row
    were written first, a document would survive here pointing at a file
    that was never stored — the failure the user sees.
    """
    before = db_session.scalar(select(Document.id).order_by(Document.created_at.desc()))

    response = client.post(
        "/api/v1/ingest/file",
        headers=auth_headers,
        files={"file": ("x.pdf", text_pdf(INVOICE), "application/pdf")},
    )

    assert response.status_code == 503
    after = db_session.scalar(select(Document.id).order_by(Document.created_at.desc()))
    assert after == before


# the citation link


def test_a_link_is_issued_for_the_owner(upload, client, auth_headers, store):
    document_id = upload().json()["document_id"]

    response = client.get(f"/api/v1/documents/{document_id}/file", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["url"].startswith("https://")
    assert response.json()["expires_in"] > 0


def test_a_link_is_not_issued_for_someone_elses_document(upload, client, db_session, store):
    """
    404 rather than 403 on purpose: a 403 confirms the id exists, which
    hands out an enumeration oracle over other people's receipts.
    """
    document_id = upload().json()["document_id"]

    other_email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    from app.core.security import hash_password
    from app.models.user import AuthIdentity, AuthProvider, User

    other = User(email=other_email, email_verified=True)
    other.identities.append(
        AuthIdentity(
            provider=AuthProvider.PASSWORD,
            provider_user_id=other_email,
            password_hash=hash_password("another-password"),
        )
    )
    db_session.add(other)
    db_session.commit()

    token = client.post(
        "/api/v1/auth/login", json={"email": other_email, "password": "another-password"}
    ).json()["access_token"]

    response = client.get(
        f"/api/v1/documents/{document_id}/file",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404

    db_session.delete(other)
    db_session.commit()


def test_a_link_for_an_unknown_document_is_404(client, auth_headers, store):
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/file", headers=auth_headers)

    assert response.status_code == 404
