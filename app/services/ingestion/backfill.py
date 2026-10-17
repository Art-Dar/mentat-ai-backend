import uuid

from sqlalchemy import Select, or_, select

from app.models import Document, IngestionStatus


def documents_needing_rechunk(
    chunk_size: int,
    chunk_overlap: int,
    *,
    include_all: bool = False,
    user_id: uuid.UUID | str | None = None,
) -> Select:
    """
    A query for the documents whose chunks are not at the target.

    COMPLETED only. A FAILED document has no usable text, and a PENDING
    one is about to be chunked by the pipeline that owns it

    `include_all` ignores the current chunking and takes every completed
    document.
    """
    query = select(Document).where(
        Document.status == IngestionStatus.COMPLETED,
        Document.content.isnot(None),
    )

    if not include_all:
        query = query.where(
            or_(
                # NULL means "ingested before the chunking was recorded",
                # which is unknown rather than correct. Treating it as
                # already-done would mark every legacy document as fine
                # and the first tuning run would measure a corpus that
                # was never re-chunked.
                Document.chunk_size.is_(None),
                Document.chunk_overlap.is_(None),
                Document.chunk_size != chunk_size,
                Document.chunk_overlap != chunk_overlap,
            )
        )

    if user_id is not None:
        query = query.where(Document.user_id == user_id)

    # Oldest first, so an interrupted run resumes in a predictable place
    # and a --limit run always advances through the library rather than
    # re-doing the same page.
    return query.order_by(Document.created_at)


__all__ = ["documents_needing_rechunk"]
