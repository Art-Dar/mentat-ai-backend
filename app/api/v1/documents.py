import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.models import Chunk, Document
from app.schemas.files import DocumentStatusResponse

router = APIRouter()


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
        document_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        user_id: uuid.UUID = Depends(get_current_user_id),
) -> DocumentStatusResponse:
    """
    Where a document got to, so a client that uploaded one can stop guessing.
    """
    document = await db.get(Document, document_id)

    # The same 404 for "not yours" as for "does not exist", matching
    # GET /documents/{id}/file. A 403 would confirm the id is real and
    # hand out an enumeration oracle over other people's receipts.
    if document is None or document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such document",
        )

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