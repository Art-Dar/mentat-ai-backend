import uuid
from datetime import datetime

from pydantic import Field, HttpUrl, field_validator
from app.schemas.base import RequestSchemal

from app.models import DocumentSource, IngestionStatus

# Source types this endpoint currently accepts. IMAGE and PDF need the media

SUPPORTED_SOURCES = frozenset(
    {DocumentSource.ARTICLE, DocumentSource.SELECTION, DocumentSource.NOTE}
)

MAX_TEXT_LENGTH = 1_000_000


class IngestRequest(RequestSchemal):
    source: DocumentSource
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    url: HttpUrl | None = None
    title: str | None = Field(default=None, max_length=512)
    published_at: datetime | None = None

    @field_validator("source")
    @classmethod
    def source_must_be_supported(cls, value: DocumentSource) -> DocumentSource:
        if value not in SUPPORTED_SOURCES:
            supported = ", ".join(sorted(s.value for s in SUPPORTED_SOURCES))
            raise ValueError(
                f"source '{value.value}' is not supported yet; use one of: {supported}"
            )
        return value


class IngestResponse(RequestSchemal):
    document_id: uuid.UUID
    status: IngestionStatus
