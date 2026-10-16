import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models import Document, TagAssignedBy
from app.schemas.tags import (
    AttachTagsRequest,
    DocumentTagRead,
    RenameTagRequest,
    TagDeleteResponse,
    TagRead,
    TagRenameResponse,
    TagWithCount,
)
from app.services.tagging.delete import TagInUse, delete_tag
from app.services.tagging.rename import TagNotFound, rename_tag
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

router = APIRouter()


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


@router.patch("/tags/{tag_id}", response_model=TagRenameResponse)
async def rename_vocabulary_tag(
    tag_id: uuid.UUID,
    payload: RenameTagRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_user),
) -> TagRenameResponse:
    """
    Rename a tag across the whole vocabulary, merging it into an existing
    tag when the new name is already taken.
    """
    try:
        result = await rename_tag(db, tag_id, payload.name)
    # TagNotFound before TagError: it is a subclass, so the order decides
    # whether a missing tag is a 404 or a 422.
    except TagNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found",
        ) from exc
    except TagError as exc:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=str(exc),
        ) from exc

    await db.commit()
    # Readable after the commit because the sessionmaker sets
    # expire_on_commit=False; otherwise this would re-query per attribute.
    return TagRenameResponse(
        tag=TagRead.model_validate(result.tag),
        merged_into=result.merged_into,
        documents_affected=result.documents_affected,
    )


@router.delete("/tags/{tag_id}", response_model=TagDeleteResponse)
async def delete_vocabulary_tag(
    tag_id: uuid.UUID,
    force: bool = Query(
        default=False,
        description=(
            "Delete even when the tag is still on documents, removing it "
            "from all of them. Without this a tag in use is refused with 409."
        ),
    ),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_user),
) -> TagDeleteResponse:
    try:
        result = await delete_tag(db, tag_id, force=force)
    # TagNotFound and TagInUse are both TagError subclasses, so these
    # three clauses have to stay in this order: specific before general,
    # or every failure collapses into one status code.
    except TagNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found",
        ) from exc
    except TagInUse as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except TagError as exc:
        raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc

    await db.commit()
    return TagDeleteResponse(
        tag_id=result.tag_id,
        name=result.name,
        documents_affected=result.documents_affected,
    )


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
    payload: AttachTagsRequest,
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
