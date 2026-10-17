import datetime
import uuid

from pydantic import BaseModel, ConfigDict

from app.models import DocumentSource, IngestionStatus

STRICT = ConfigDict(extra="forbid")


class FileIngestResponse(BaseModel):
    """
    status is PENDING because the response is built before the background
    task runs — the same contract /ingest already has.
    """

    model_config = STRICT

    document_id: uuid.UUID
    status: IngestionStatus
    page_count: int
    pages_needing_ocr: list[int]
    needs_ocr: bool
    ocr_pages: list[int] = []
    ocr_confidence: float | None = None


class DocumentStatusResponse(BaseModel):
    """
    Where a document got to, for a client that uploaded one and is waiting.
    """

    model_config = STRICT

    document_id: uuid.UUID
    source: DocumentSource
    status: IngestionStatus
    title: str | None
    error_message: str | None
    char_count: int
    chunk_count: int
    page_count: int
    has_file: bool
    language: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @property
    def is_searchable(self) -> bool:  # pragma: no cover - convenience only
        return self.chunk_count > 0


class ImageIngestResponse(BaseModel):
    """
    The OCR counterpart to FileIngestResponse.
    """

    model_config = STRICT

    document_id: uuid.UUID
    status: IngestionStatus
    char_count: int
    word_count: int
    mean_confidence: float | None
    low_confidence: bool
    language: str


class FileLinkResponse(BaseModel):
    """
    A short-lived download URL for the stored original
    `expires_in` is returned so the client can cache the URL
    """

    model_config = STRICT

    url: str
    # `expires_in` is returned so the client can cache the URL — and so
    # it knows to re-fetch on click rather than at render time.
    expires_in: int
    media_type: str
    # The name the user uploaded, for a download affordance to label
    filename: str | None = None
    page_count: int = 0
