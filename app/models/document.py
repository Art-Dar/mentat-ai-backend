import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.user import User


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """Persist enum *values* (lowercase) rather than member names."""
    return [member.value for member in enum_cls]


class DocumentSource(enum.StrEnum):
    ARTICLE = "article"  # full page via Readability
    SELECTION = "selection"  # highlighted text, via the Selection API
    IMAGE = "image"  # OCR
    PDF = "pdf"
    NOTE = "note"


class IngestionStatus(enum.StrEnum):
    PENDING = "pending"  # row created, background task not started
    PROCESSING = "processing"  # normalize → chunk → embed → tag in flight
    COMPLETED = "completed"  # chunks and embeddings exist, searchable
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_created", "user_id", "created_at"),
        Index("ix_documents_user_content_hash", "user_id", "content_hash"),
        Index("ix_documents_url", "url"),
        CheckConstraint(
            "(status = 'failed') = (error_message IS NOT NULL)",
            name="ck_documents_error_message_iff_failed",
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[DocumentSource] = mapped_column(
        SAEnum(DocumentSource, name="document_source", values_callable=_enum_values)
    )
    status: Mapped[IngestionStatus] = mapped_column(
        SAEnum(IngestionStatus, name="ingestion_status", values_callable=_enum_values),
        default=IngestionStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(2048))
    title: Mapped[str | None] = mapped_column(String(512))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    file_path: Mapped[str | None] = mapped_column(String(1024))
    content: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8))
    content_hash: Mapped[str | None] = mapped_column(String(64))

    user: Mapped["User"] = relationship(back_populates="documents")
