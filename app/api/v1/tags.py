import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models import Document, TagAssignedBy
from app.schemas.tags import AttachTagsRequestAttachTagsRequest, DocumentTagRead, TagRead, TagWithCount
from app.services.tagging.tags import (
    TagError,
    attach_tags,
    detach_tag,
    list_tags,
    tags_for_document,
)

# Spelled as a number rather than status.HTTP_422_*: Starlette renamed that
# constant and the old name is deprecated while the new one is missing from
# older versions. 422 is neither.
_UNPROCESSABLE = 422

router = APIRouter(prefix="/tags")


async def _require_document(db: AsyncSession, document_id: uuid.UUID) -> Document:
    """
    404 for a document that does not exist
    """
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return document


def _as_links(rows) -> list[DocumentTagRead]:
    return [
        DocumentTagRead(
            tag=TagRead.model_validate(link.tag),
            assigned_by=link.assigned_by,
            confidence=link.confidence,
        )
        for link in rows
    ]


@router.get("/tags", response_model=list[TagWithCount])
async def read_tags(
        db: AsyncSession = Depends(get_db),
        _: str = Depends(get_current_user),
) -> list[TagWithCount]:
    """Every tag with its usage count, most used first."""
    return [
        TagWithCount(id=tag.id, name=tag.name, color=tag.color, document_count=count)
        for tag, count in await list_tags(db)
    ]


@router.get("/documents/{document_id}/tags", response_model=list[DocumentTagRead])
async def read_document_tags(
        document_id: uuid.UUID,
        db: AsyncSession = Depends(get_db),
        _: str = Depends(get_current_user),
) -> list[DocumentTagRead]:
    await _require_document(db, document_id)
    return _as_links(await tags_for_document(db, document_id))


@router.post(
    "/documents/{document_id}/tags",
    response_model=list[DocumentTagRead],
    status_code=status.HTTP_201_CREATED,
)
async def add_document_tags(
        document_id: uuid.UUID,
        payload: AttachTagsRequestAttachTagsRequest,
        db: AsyncSession = Depends(get_db),
        _: str = Depends(get_current_user),
) -> list[DocumentTagRead]:
    """
    Attach tags, creating any that do not exist yet.
    Idempotent: posting a tag the document already has is a 201 with the
    unchanged list, not a conflict.
    """
    await _require_document(db, document_id)

    try:
        await attach_tags(
            db,
            document_id,
            payload.names,
            assigned_by=TagAssignedBy.USER,
        )
    except TagError as exc:
        # A bad name or colour is the caller's mistake, not a server fault.
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=str(exc),
        ) from exc

    await db.commit()
    return _as_links(await tags_for_document(db, document_id))


@router.delete("/documents/{document_id}/tags", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document_tag(
        document_id: uuid.UUID,
        name: str = Query(min_length=1, description="Tag name; matched case-insensitively"),
        db: AsyncSession = Depends(get_db),
        _: str = Depends(get_current_user),
) -> Response:
    """
    Detach one tag. The Tag row itself survives — it is probably on other
    documents

    The name is a query parameter rather than a path segment because tag
    names may contain a slash ("ml/nlp")
    """
    await _require_document(db, document_id)

    try:
        removed = await detach_tag(db, document_id, name)
    except TagError as exc:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=str(exc),
        ) from exc

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That tag is not on this document",
        )

    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
