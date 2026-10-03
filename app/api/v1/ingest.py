from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models import User, Document
from app.schemas.ingest import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("", status_code=status.HTTP_201_CREATED, response_model=IngestResponse)
async def ingest(
    payload: IngestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    # accept a capture and return immediately with its document ID
    # row is created with status=PENDING and the raw text in `content`
    # (normalize -> chunk -> embed -> tag) by background worker, until - pending

    """HANDOFF: the queue ticket adds exactly one call here, after the commit:
        background_tasks.add_task(process_document, document.id)"""

    document = Document(
        user_id=current_user.id,
        source=payload.source,
        url=str(payload.url) if payload.url else None,
        title=payload.title,
        published_at=payload.published_at,
        # raw capture - the pipeline replaces this with the normalized text.
        content=payload.text,
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    return IngestResponse(document_id=document.id, status=document.status)