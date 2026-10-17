# after /ingest has already responded - each stage is added
# (normalize -> chunk -> embed -> tag)
# here as its ticket lands, the endpoint never changes

import logging
import uuid
from hashlib import sha256

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentSource, DocumentTag, IngestionStatus, Tag
from app.services.ingestion.chunking import chunk_text
from app.services.ingestion.embedder import Embedder, get_embedder
from app.services.ingestion.normalization import normalize_text
from app.services.tagging.auto_tagger import VOCABULARY_LIMIT, Tagger, get_tagger
from app.services.tagging.tags import attach_tags, replace_auto_tags

logger = logging.getLogger(__name__)

# maps how the user captured something to how the text should be normalized.
_NORMALIZER_SOURCE = {
    DocumentSource.ARTICLE: "readability",
    DocumentSource.SELECTION: "selection",
    DocumentSource.NOTE: "selection",
    DocumentSource.PDF: "readability",
    DocumentSource.IMAGE: "ocr",
}


async def process_document(
    document_id: uuid.UUID,
    embedder: Embedder | None = None,
    tagger: Tagger | None = None,
) -> None:
    # normalize -> chunk and embed document

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            logger.warning("process_document: %s no longer exists", document_id)
            return

        document.status = IngestionStatus.PROCESSING
        document.error_message = None
        await db.commit()

        try:
            _normalize(document)
            await rechunk_and_embed(db, document, embedder or get_embedder())
            await _assign_tags(document, db, tagger)
            document.status = IngestionStatus.COMPLETED
            document.error_message = None
        except Exception as exc:
            logger.exception("ingestion failed for document %s", document_id)
            await _record_failure(db, document_id, exc)

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


async def rechunk_and_embed(
    db: AsyncSession,
    document: Document,
    embedder: Embedder,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Replace a document's chunks and their vectors.
    """
    chunks = await _rechunk(document, db, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    await _embed(chunks, embedder)
    return chunks


async def _rechunk(
    document: Document,
    db: AsyncSession,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Replace this document's chunks. Embeddings stay NULL until the
    embedding stage fills them.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    await db.execute(delete(Chunk).where(Chunk.document_id == document.id))

    chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=piece.index,
            content=piece.text,
            start_char=piece.start_char,
            end_char=piece.end_char,
        )
        for piece in chunk_text(document.content or "", chunk_size=size, overlap=overlap)
    ]
    db.add_all(chunks)

    document.chunk_size = size
    document.chunk_overlap = overlap
    return chunks


async def _embed(chunks: list[Chunk], embedder: Embedder) -> None:
    # fill in `embedding` and `embedding_model` for every chunk
    # batched call for the whole document rather than one call per chunk
    if not chunks:
        # Normalization rejects text too short to be useful, so this means an
        # edge case rather than an empty document — worth a log, not a crash.
        logger.info("no chunks to embed")
        return

    vectors = await embedder.embed_documents([chunk.content for chunk in chunks])

    # strict=True is safe here and wanted: embed_documents guarantees one
    # vector per input, so a mismatch is a bug in the embedder, not data.
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
        chunk.embedding_model = settings.embedding_model


async def _record_failure(db: AsyncSession, document_id: uuid.UUID, exc: Exception) -> None:
    """
    Roll back the half-finished work, then mark the document FAILED.
    Without it, a document whose embedding call times out would commit
    its chunks with NULL vectors
    """
    await db.rollback()

    document = await db.get(Document, document_id)
    if document is None:  # deleted while we were working — nothing to record
        return

    document.status = IngestionStatus.FAILED
    document.error_message = str(exc)[:1000]
    await db.commit()


async def _popular_tag_names(db, limit=VOCABULARY_LIMIT) -> list[str]:
    # The most-used existing tags, offered to Claude as reuse candidates.
    result = await db.execute(
        select(Tag.name)
        .outerjoin(DocumentTag, DocumentTag.tag_id == Tag.id)
        .group_by(Tag.id)
        .order_by(func.count(DocumentTag.document_id).desc(), func.lower(Tag.name))
        .limit(limit)
    )
    return list(result.scalars().all())


async def _assign_tags(document, db, tagger=None) -> None:
    # ask llm for topic tags and store them as AUTO assignments.
    try:
        async with db.begin_nested():
            resolved = tagger or get_tagger()
            vocabulary = await _popular_tag_names(db)

            suggestions = await resolved.suggest_tags(
                title=document.title,
                text=document.content or "",
                vocabulary=vocabulary,
            )

            # Clear the previous pass's guesses, keeping anything the user
            # applied by hand, then attach each suggestion with its own
            # score — replace_auto_tags takes a single confidence, which is
            # why the two steps are separate here.
            await replace_auto_tags(db, document.id, [])
            for suggestion in suggestions:
                await attach_tags(
                    db,
                    document.id,
                    [suggestion.name],
                    confidence=suggestion.confidence,
                )

        logger.info("tagged document %s with %d tags", document.id, len(suggestions))
    except Exception:
        logger.warning(
            "auto-tagging failed for document %s; it keeps its embeddings and "
            "can be re-tagged later",
            document.id,
            exc_info=True,
        )
