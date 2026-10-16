import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models import IngestionStatus, DocumentSource


class FileIngestResponse(BaseModel):
    """
    status is PENDING because the response is built before the background
    task runs — the same contract /ingest already has.
    """

    document_id: uuid.UUID
    status: IngestionStatus
    page_count: int
    pages_needing_ocr: list[int]
    needs_ocr: bool


class FileLinkResponse(BaseModel):
    """
    A short-lived download URL for the stored original
    `expires_in` is returned so the client can cache the URL
    """

    url: str
    expires_in: int


class ImageIngestResponse(BaseModel):
    """
    The OCR counterpart to FileIngestResponse.
    """

    document_id: uuid.UUID
    status: IngestionStatus
    char_count: int
    word_count: int
    mean_confidence: float | None
    low_confidence: bool
    language: str


class DocumentStatusResponse(BaseModel):
    """
    Where a document got to, for a client that uploaded one and is waiting.
    """

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