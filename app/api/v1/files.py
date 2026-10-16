import logging
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models import Document, DocumentSource, IngestionStatus
from app.schemas.files import FileIngestResponse, FileLinkResponse
from app.services.ingestion.extractors.pdf import (
    MAX_PDF_BYTES,
    EncryptedPdfError,
    PdfExtractionError,
    extract_pdf_text_async,
)
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


def _caller_id(subject: str) -> uuid.UUID:
    """
    The authenticated caller's user_id
    """
    try:
        return uuid.UUID(subject)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        ) from exc


# Read in pieces so a hostile Content-Length cannot be trusted and a large
# body is refused before it is all in memory.
_CHUNK = 1024 * 1024

# Every PDF starts with this. The client's Content-Type is a header it
# chose, so it proves nothing; the first five bytes are the file itself.
_PDF_MAGIC = b"%PDF-"

_UNPROCESSABLE = 422
_TOO_LARGE = 413
_UNSUPPORTED_MEDIA = 415


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
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
    store: FileStore = Depends(get_file_store),
) -> FileIngestResponse:
    """
    Accept a PDF, extract its text, keep the original, queue the pipeline
    """
    user_id = _caller_id(current_user)

    data = await _read_capped(file, MAX_PDF_BYTES)

    if not data.startswith(_PDF_MAGIC):
        # 415 rather than 422: the request is well-formed, the media type
        # is the problem. An image goes to the OCR endpoint, not this one.
        raise HTTPException(
            status_code=_UNSUPPORTED_MEDIA,
            detail="Only PDF files are accepted here",
        )

    try:
        extraction = await extract_pdf_text_async(data)
    except EncryptedPdfError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc
    except PdfExtractionError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc

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
                "needs_ocr": page.needs_ocr,
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
        pages_needing_ocr=list(extraction.pages_needing_ocr),
        needs_ocr=extraction.needs_ocr,
    )


@router.get("/documents/{document_id}/file", response_model=FileLinkResponse)
async def get_document_file_link(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
    store: FileStore = Depends(get_file_store),
) -> FileLinkResponse:
    """
    A short-lived link to the original file, for a citation to point at.

    Returns a URL rather than the bytes. A presigned URL is served by R2
    directly, so a worker is not tied up streaming a 20 MB scan — and the
    authorization decision still happens here, before the URL exists.
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
    if document.user_id != _caller_id(current_user):
        raise missing

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

    return FileLinkResponse(url=url, expires_in=DEFAULT_URL_TTL_SECONDS)
