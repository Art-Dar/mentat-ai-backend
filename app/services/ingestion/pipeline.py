# after /ingest has already responded - each stage is added
# (normalize -> chunk -> embed -> tag)
# here as its ticket lands, the endpoint never changes

import logging
import uuid
from hashlib import sha256

from app.core.db import AsyncSessionLocal
from app.models import Document, DocumentSource, IngestionStatus
from app.services.ingestion.normalization import normalize_text
from app.models import Chunk
from app.services.ingestion.chunking import chunk_text

logger = logging.getLogger(__name__)

# maps how the user captured something to how the text should be normalized.
_NORMALIZER_SOURCE = {
    DocumentSource.ARTICLE: "readability",
    DocumentSource.SELECTION: "selection",
    DocumentSource.NOTE: "selection",
    DocumentSource.PDF: "readability",
    DocumentSource.IMAGE: "ocr",
}


async def process_document(document_id: uuid.UUID) -> None:
    # normalize -> chunk and embed document

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            logger.warning("process_document: %s no longer exists", document_id)
            return

        document.status = IngestionStatus.PROCESSING
        await db.commit()

        try:
            _normalize(document)
            await _rechunk(document, db)
            # embed_chunks(chunks)                               <- SB 44
            # assign_tags(document)                              <- Story 5.1
            document.status = IngestionStatus.COMPLETED
            document.error_message = None
        except Exception as exc:
            logger.exception("ingestion failed for document %s", document_id)
            document.status = IngestionStatus.FAILED
            document.error_message = str(exc)[:1000]

        await db.commit()


def _normalize(document: Document) -> None:
    source_type = _NORMALIZER_SOURCE.get(document.source, "readability")
    result = normalize_text(document.content or "", source_type=source_type)

    if not result.is_likely_valid:
        raise ValueError(
            f"Extracted text too short to be useful "
            f"({result.normalized_char_count} chars) — likely an extraction failure"
        )

    document.content = result.text
    document.language = result.detected_language
    document.content_hash = sha256(result.text.encode("utf-8")).hexdigest()

async def _rechunk(document: Document, db: AsyncSession) -> None:
    """Replace this document's chunks. Embeddings stay NULL until the
    embedding stage fills them."""
    await db.execute(delete(Chunk).where(Chunk.document_id == document.id))

    for piece in chunk_text(document.content or ""):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=piece.index,
                content=piece.text,
                start_char=piece.start_char,
                end_char=piece.end_char,
            )
        )