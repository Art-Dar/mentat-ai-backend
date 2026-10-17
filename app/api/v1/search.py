import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.schemas.search import SearchHitRead, SearchRequest, SearchResponse
from app.services.ingestion.embedder import Embedder, get_embedder
from app.services.retrieval.search import search_by_text

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    # Injected rather than resolved inside search_by_text, for the same
    # dependency_overrides instead of monkeypatching a module global, and
    # the suite never calls Voyage or pays for a query embedding.
    embedder: Embedder = Depends(get_embedder),
) -> SearchResponse:
    """
    Find the passages in your own documents closest to a question.
    The question is embedded with the embedder's QUERY input type.
    """
    try:
        hits = await search_by_text(
            db,
            payload.query,
            user_id=user_id,
            embedder=embedder,
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
            document_ids=payload.document_ids,
        )
    except ValueError as exc:
        # The service validates too, and disagreement between the two
        # would otherwise surface as a 500. A dimension mismatch means the
        # embedder is misconfigured — the server's fault, not the query's.
        logger.exception("retrieval rejected a query it should have accepted")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is misconfigured on the server",
        ) from exc

    return SearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        count=len(hits),
        hits=[
            SearchHitRead(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                chunk_index=hit.chunk_index,
                content=hit.content,
                start_char=hit.start_char,
                end_char=hit.end_char,
                similarity=hit.similarity,
                document_title=hit.document_title,
                document_url=hit.document_url,
            )
            for hit in hits
        ],
    )