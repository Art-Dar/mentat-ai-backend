import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.models import Chunk, Document
from app.schemas.files import DocumentStatusResponse
from app.schemas.passage import PassageResponse
from app.services.retrieval.passage import (
    MAX_WINDOW_CHARS,
    WINDOW_CHARS,
    PassageUnavailable,
    extract_passage,
)

router = APIRouter()


async def _own_document(db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    """
    The caller's own document, or a 404.

    The same 404 for "not yours" as for "does not exist", matching
    GET /documents/{id}/file. A 403 would confirm the id is real and hand
    out an enumeration oracle over other people's receipts.
    """
    document = await db.get(Document, document_id)
    if document is None or document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such document",
        )
    return document


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> DocumentStatusResponse:
    """
    Where a document got to, so a client that uploaded one can stop guessing.
    """
    document = await _own_document(db, document_id, user_id)

    chunk_count = (
        await db.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
        )
    ).scalar_one()

    return DocumentStatusResponse(
        document_id=document.id,
        source=document.source,
        status=document.status,
        title=document.title,
        # Populated by the pipeline when ingestion fails. It is the only
        # place the reason is recorded, so it has to be reachable.
        error_message=document.error_message,
        char_count=len(document.content or ""),
        chunk_count=chunk_count,
        page_count=len(document.page_map or []),
        has_file=document.storage_key is not None,
        language=document.language,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get("/documents/{document_id}/passage", response_model=PassageResponse)
async def get_document_passage(
    document_id: uuid.UUID,
    start: int = Query(ge=0, description="start_char from the citation"),
    end: int = Query(ge=1, description="end_char from the citation"),
    window: int = Query(
        default=WINDOW_CHARS,
        ge=0,
        le=MAX_WINDOW_CHARS,
        description="characters of surrounding document to include either side",
    ),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> PassageResponse:
    """
    Read a cited passage inside its surrounding text.

    This is the endpoint that makes a citation checkable. The client
    passes a citation's `start_char` and `end_char` straight through and
    gets the passage back with a window of document either side and the
    offsets to highlight
    """
    document = await _own_document(db, document_id, user_id)

    try:
        passage = extract_passage(
            document.content,
            start,
            end,
            window=window,
            page_map=document.page_map,
        )
    except PassageUnavailable as exc:
        # 409, not 422. The request is well-formed; the document moved
        # underneath it — almost always a re-ingest between the search and
        # this call. A client that sees 409 should re-run the search,
        # which is a different recovery from fixing its own parameters.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return PassageResponse.from_passage(
        passage,
        document_id=document.id,
        title=document.title,
        source=document.source,
    )
