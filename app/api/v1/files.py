import logging
import re
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.models import Document, DocumentSource, IngestionStatus
from app.schemas.files import FileIngestResponse, FileLinkResponse, ImageIngestResponse
from app.services.ingestion.extractors.ocr import (
    DEFAULT_LANGUAGE,
    MAX_IMAGE_BYTES,
    MIN_CHARS,
    OcrEngineUnavailable,
    OcrError,
    extract_image_text_async,
)
from app.services.ingestion.extractors.pdf import (
    MAX_PDF_BYTES,
    EncryptedPdfError,
    PdfExtractionError,
)
from app.services.ingestion.extractors.scanned_pdf import extract_pdf_text_with_ocr_async
from app.services.ingestion.normalization import MIN_VALID_LENGTH
from app.services.ingestion.scheduler import schedule_processing
from app.services.storage.base import (
    DEFAULT_URL_TTL_SECONDS,
    FileStore,
    ObjectNotFound,
    StorageError,
    storage_key,
)
from app.services.storage.s3_store import get_file_store

logger = logging.getLogger(__name__)

router = APIRouter()

# Read in pieces so a hostile Content-Length cannot be trusted and a large
# body is refused before it is all in memory.
_CHUNK = 1024 * 1024

# Every PDF starts with this. The client's Content-Type is a header it
# chose, so it proves nothing; the first five bytes are the file itself.
_PDF_MAGIC = b"%PDF-"

_UNPROCESSABLE = 422
_TOO_LARGE = 413
_UNSUPPORTED_MEDIA = 415

_IMAGE_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
)
_WEBP = ("webp", "image/webp")
_ACCEPTED_IMAGES = "PNG, JPEG, WEBP or TIFF"

# Built from the same table the upload path matches magic bytes against,
# so the media type we hand back on download cannot drift from the one we
# stored. The storage key's extension is the record of what was detected:
# it was derived from the file's own bytes, never from the request.
_MEDIA_TYPE_BY_EXTENSION: dict[str, str] = {
    extension: media_type for _, extension, media_type in _IMAGE_MAGIC
}
_MEDIA_TYPE_BY_EXTENSION[_WEBP[0]] = _WEBP[1]
_MEDIA_TYPE_BY_EXTENSION["pdf"] = "application/pdf"

_PDF_MEDIA_TYPE = "application/pdf"

# Tesseract language codes
# three letters, joined by + for multi-language pages ("eng+ukr")
_LANGUAGE_RE = re.compile(r"^[a-z]{3}(\+[a-z]{3})*$")


def _check_language(language: str) -> None:
    if not _LANGUAGE_RE.match(language):
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="language must be a tesseract code such as 'eng' or 'eng+ukr'",
        )


def _image_format(data: bytes) -> tuple[str, str] | None:
    """The file's real format, or None if it is not one we accept."""
    for magic, extension, media_type in _IMAGE_MAGIC:
        if data.startswith(magic):
            return extension, media_type
    # WEBP is a RIFF container: the format marker sits at offset 8, so it
    # cannot be matched by a prefix like the others.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _WEBP
    return None


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    """
    Read the upload, refusing anything over the limit mid-stream.

    Reading first and checking the length afterwards would mean a 2 GB
    upload is fully buffered before being rejected — which is the attack,
    not the protection.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=_TOO_LARGE,
                detail=f"File exceeds the {limit // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/ingest/file",
    response_model=FileIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    language: str = Form(default=DEFAULT_LANGUAGE),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    store: FileStore = Depends(get_file_store),
) -> FileIngestResponse:
    """
    Accept a PDF, extract its text, keep the original, queue the pipeline
    """
    _check_language(language)

    data = await _read_capped(file, MAX_PDF_BYTES)

    if not data.startswith(_PDF_MAGIC):
        # 415 rather than 422: the request is well-formed, the media type
        # is the problem. An image goes to the OCR endpoint, not this one.
        raise HTTPException(
            status_code=_UNSUPPORTED_MEDIA,
            detail="Only PDF files are accepted here",
        )

    try:
        # OCR-capable: a PDF with a complete text layer never starts the
        # engine, so this costs nothing for the common case and is the
        # difference between working and silently failing for a scan.
        ocr_result = await extract_pdf_text_with_ocr_async(data, language=language)
    except EncryptedPdfError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc
    except PdfExtractionError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc

    extraction = ocr_result.extraction

    if len(extraction.text.strip()) < MIN_VALID_LENGTH:
        if not ocr_result.ocr_available:
            # We never got to look. Saying "this PDF has no readable text"
            # would be a claim the server cannot support, and it would send
            # the user off to re-scan a document that is probably fine.
            logger.error("OCR unavailable and the text layer was empty")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "This PDF has no text layer and text recognition is "
                    "unavailable on the server, so it could not be read. "
                    "Please retry later."
                ),
            )
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=(
                "No readable text could be extracted from this PDF, even "
                "after scanning it for text in images. If it is a scan, try "
                "a sharper copy; if it is a form, check it is not empty."
            ),
        )

    # The id is generated here, not by the database, so the storage key
    # exists before the row does — which is what lets the bytes be written
    # first.
    document_id = uuid.uuid4()
    key = storage_key(user_id, document_id)

    try:
        await store.put(key, data, content_type="application/pdf")
    except StorageError as exc:
        logger.exception("could not store %s", key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage is unavailable, please retry",
        ) from exc

    document = Document(
        id=document_id,
        user_id=user_id,
        source=DocumentSource.PDF,
        content=extraction.text,
        title=title or file.filename,
        status=IngestionStatus.PENDING,
        storage_key=key,
        file_size=len(data),
        # Kept for display only. It never reaches a path — see storage_key.
        original_filename=file.filename,
        page_map=[
            {
                "index": page.index,
                "start_char": page.start_char,
                "end_char": page.end_char,
                # "still has no usable text", matching the response's
                # pages_needing_ocr. NOT PageText.needs_ocr
                "needs_ocr": page.index in ocr_result.still_needs_ocr,
                "source": "ocr" if page.index in ocr_result.ocr_pages else "pdf",
            }
            for page in extraction.pages
        ],
    )
    db.add(document)

    try:
        await db.commit()
    except Exception:
        # The row lost; the object would otherwise linger with nothing
        # pointing at it. Best effort — if this also fails the sweeper gets
        # it, which is exactly why bytes-first is the safe ordering.
        logger.exception("commit failed after storing %s, removing the object", key)
        try:
            await store.delete(key)
        except StorageError:
            logger.exception("could not remove orphaned object %s", key)
        raise

    await schedule_processing(document.id, background_tasks)

    return FileIngestResponse(
        document_id=document.id,
        status=document.status,
        page_count=extraction.page_count,
        # After the OCR pass, so these are pages still without text —
        # over the page budget, or genuinely unreadable.
        pages_needing_ocr=list(ocr_result.still_needs_ocr),
        needs_ocr=bool(ocr_result.still_needs_ocr),
        ocr_pages=list(ocr_result.ocr_pages),
        ocr_confidence=ocr_result.mean_confidence,
    )


@router.post(
    "/ingest/image",
    response_model=ImageIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    language: str = Form(default=DEFAULT_LANGUAGE),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    store: FileStore = Depends(get_file_store),
) -> ImageIngestResponse:
    """
    Accept a scan or photo, OCR it, keep the original, queue the pipeline.

    The stored original is the point, not a side effect: OCR is lossy and
    improves between tesseract versions, so the bytes are what a later pass
    re-reads and what a citation points a user at. Retrieval is the same
    GET /documents/{id}/file the PDF path uses — the object store does not
    care what is in the object.
    """
    _check_language(language)

    data = await _read_capped(file, MAX_IMAGE_BYTES)

    image_format = _image_format(data)
    if image_format is None:
        raise HTTPException(
            status_code=_UNSUPPORTED_MEDIA,
            detail=f"Only {_ACCEPTED_IMAGES} images are accepted here",
        )
    extension, media_type = image_format

    try:
        extraction = await extract_image_text_async(data, language=language)
    except OcrEngineUnavailable as exc:
        # The upload was fine; the host is missing its tesseract binary or
        # the engine hung. 503, not 422 — a retry may well succeed, and the
        # user has nothing to fix.
        logger.exception("OCR engine unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text recognition is unavailable, please retry",
        ) from exc
    except OcrError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc

    if not extraction.has_text:
        # Refused before anything is written. A document with no content
        # would be invisible in search and indistinguishable from a failed
        # upload, so it is better to tell the user now while they still
        # have the receipt in their hand.
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=(
                f"No readable text found in this image "
                f"(under {MIN_CHARS} characters). Try a sharper, "
                f"straight-on photo in better light."
            ),
        )

    document_id = uuid.uuid4()
    key = storage_key(user_id, document_id, extension=extension)

    try:
        await store.put(key, data, content_type=media_type)
    except StorageError as exc:
        logger.exception("could not store %s", key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage is unavailable, please retry",
        ) from exc

    document = Document(
        id=document_id,
        user_id=user_id,
        source=DocumentSource.IMAGE,
        content=extraction.text,
        title=title or file.filename,
        status=IngestionStatus.PENDING,
        storage_key=key,
        file_size=len(data),
        original_filename=file.filename,
        # One entry, so the column means the same thing for both sources:
        # which span of `content` came from where. needs_ocr is False
        # because this text IS the OCR result — there is no second pass to
        # queue.
        #
        # The per-word bounding boxes stay out of the database on purpose:
        # a dense receipt is several hundred words, and nothing reads them
        # yet. When the citation epic needs them they belong in their own
        # table, not inflating every document row.
        page_map=[
            {
                "index": 0,
                "start_char": 0,
                "end_char": len(extraction.text),
                "needs_ocr": False,
                "source": "ocr",
                "language": extraction.language,
                "word_count": extraction.word_count,
                "mean_confidence": extraction.mean_confidence,
            }
        ],
    )
    db.add(document)

    try:
        await db.commit()
    except Exception:
        logger.exception("commit failed after storing %s, removing the object", key)
        try:
            await store.delete(key)
        except StorageError:
            logger.exception("could not remove orphaned object %s", key)
        raise

    await schedule_processing(document.id, background_tasks)

    return ImageIngestResponse(
        document_id=document.id,
        status=document.status,
        char_count=extraction.char_count,
        word_count=extraction.word_count,
        mean_confidence=extraction.mean_confidence,
        low_confidence=extraction.is_low_confidence,
        language=extraction.language,
    )


def _media_type_for(storage_key_value: str) -> str:
    """
    Falls back to application/octet-stream rather than guessing: a client
    that cannot tell what it received will offer a download, which is the
    safe wrong answer. Claiming application/pdf for something else would
    be the unsafe one.
    """
    _, _, extension = storage_key_value.rpartition(".")
    return _MEDIA_TYPE_BY_EXTENSION.get(extension.lower(), "application/octet-stream")


@router.get("/documents/{document_id}/file", response_model=FileLinkResponse)
async def get_document_file_link(
    document_id: uuid.UUID,
    page: int | None = Query(
        default=None,
        ge=1,
        description="open a PDF at this 1-based page, e.g. citation.pages[0]",
    ),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    store: FileStore = Depends(get_file_store),
) -> FileLinkResponse:
    """
    A short-lived link to the original file, for a citation to point at.

    Returns a URL rather than the bytes. A presigned URL is served by R2
    directly, so a worker is not tied up streaming a 20 MB scan — and the
    authorization decision still happens here, before the URL exists.

    With `page`, the URL carries a `#page=N` fragment, which opens a PDF
    at that page in every built-in browser viewer. Without it the user
    lands on page 1 of a forty-page invoice holding a citation that says
    "page 31" — a reference that points at the file rather than into it,
    which is the thing this story exists to fix.
    """
    missing = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No file for that document",
    )

    document = await db.get(Document, document_id)
    if document is None or document.storage_key is None:
        raise missing

    # Ownership, not merely authentication. Without this any signed-in user
    # could read any other user's receipts by guessing a document id.
    #
    # The same 404 as "it does not exist", deliberately: a 403 would
    # confirm the id is real, handing out an enumeration oracle over other
    # people's receipts.
    if document.user_id != user_id:
        raise missing

    media_type = _media_type_for(document.storage_key)
    page_count = len(document.page_map or [])

    if page is not None and page_count and page > page_count:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=f"This document has {page_count} page(s); page {page} does not exist",
        )

    try:
        url = await store.url_for(document.storage_key, expires_in=DEFAULT_URL_TTL_SECONDS)
    except ObjectNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="The stored file is missing"
        ) from exc
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage is unavailable",
        ) from exc

    # Appended after signing, deliberately. A fragment is never sent to
    # the server, so it is outside the signature by definition and adding
    # it cannot invalidate one. It is also dropped for anything but a PDF
    # rather than rejected
    if page is not None and media_type == _PDF_MEDIA_TYPE:
        url = f"{url}#page={page}"

    return FileLinkResponse(
        url=url,
        expires_in=DEFAULT_URL_TTL_SECONDS,
        media_type=media_type,
        filename=document.original_filename,
        page_count=page_count,
    )
