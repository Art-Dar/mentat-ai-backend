import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 100

# pgvector's HNSW search -- default of 40 caps how many candidates
_DEFAULT_EF_SEARCH = 40


@dataclass(frozen=True)
class SearchHit:
    """
    One retrieved chunk, with everything a citation needs.
    Carries the document's title and URL so the caller does not have to
    issue a second query per hit just to render a result.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    similarity: float
    document_title: str | None
    document_url: str | None


def _build_query(
    query_vector: list[float],
    top_k: int,
    document_ids: Sequence[uuid.UUID] | None,
) -> Select:
    # .label() so the same expression can be reused in ORDER BY without
    # Postgres computing the distance twice per row.
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

    statement = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.content,
            Chunk.start_char,
            Chunk.end_char,
            distance,
            Document.title,
            Document.url,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )

    if document_ids is not None:
        # An empty sequence means "search nothing", which IN () expresses
        # correctly — different from None, which means "search everything".
        statement = statement.where(Chunk.document_id.in_(document_ids))

    return statement


async def search_chunks(
    db: AsyncSession,
    query_vector: list[float],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = 0.0,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """
    Return the `top_k` chunks closest to `query_vector`, best first.
    `min_similarity` drops weak matches
    0.0 keeps everything with a non-negative score

    `document_ids` scopes the search — for "search within this article",
    or for a conversation already pinned to a few sources.
    """
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"query vector has {len(query_vector)} dimensions, expected {EMBEDDING_DIM} — "
            f"check output_dimension was passed to the embedder"
        )
    if not 0 < top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")

    if top_k > _DEFAULT_EF_SEARCH // 2:
        # SET LOCAL lasts for this transaction only, so one slow query does
        # not make every later query in the session slow too.
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {min(top_k * 4, 1000)}"))

    rows = (await db.execute(_build_query(query_vector, top_k, document_ids))).all()

    hits = [
        SearchHit(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            content=row.content,
            start_char=row.start_char,
            end_char=row.end_char,
            similarity=1.0 - float(row.distance),
            document_title=row.title,
            document_url=row.url,
        )
        for row in rows
    ]
    return [hit for hit in hits if hit.similarity >= min_similarity]


async def search_by_text(
    db: AsyncSession,
    query: str,
    *,
    embedder: Embedder | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = 0.0,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """
    Embed a question and search with it.

    `embed_query`, NOT `embed_documents` (same more robust res)
    a stored passage is long and declarative
    a question is short and interrogative
    """
    if not query.strip():
        return []

    embedder = embedder or get_embedder()
    query_vector = await embedder.embed_query(query)

    return await search_chunks(
        db,
        query_vector,
        top_k=top_k,
        min_similarity=min_similarity,
        document_ids=document_ids,
    )