import uuid

from pydantic import BaseModel

from app.models import IngestionStatus


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
