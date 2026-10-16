import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentTag, Tag
from app.services.tagging.rename import TagNotFound
from app.services.tagging.tags import TagError


class TagInUse(TagError):
    """
    The tag is still on documents, and the caller did not say to proceed
    """

    def __init__(self, name: str, document_count: int) -> None:
        self.document_count = document_count
        super().__init__(
            f"'{name}' is still on {document_count} document(s). "
            "Delete it anyway with force=true to remove it from all of them."
        )


@dataclass(frozen=True)
class TagDeletion:
    """What was removed. Returned because the caller cannot look it up after."""

    tag_id: uuid.UUID
    name: str
    documents_affected: int


async def delete_tag(db: AsyncSession, tag_id: uuid.UUID, *, force: bool = False) -> TagDeletion:
    """
    Remove a tag from the vocabulary.
    """
    tag = await db.get(Tag, tag_id)
    if tag is None:
        raise TagNotFound("Tag not found")

    # Counted rather than len(tag.documents): the relationship would load
    # every link row to measure them, and a popular tag has thousands.
    document_count = (
        await db.execute(
            select(func.count()).select_from(DocumentTag).where(DocumentTag.tag_id == tag_id)
        )
    ).scalar_one()

    if document_count and not force:
        raise TagInUse(tag.name, document_count)

    # Read the name before the delete: afterwards the instance is expired
    # and touching it would re-query a row that no longer exists.
    name = tag.name

    await db.delete(tag)

    return TagDeletion(tag_id=tag_id, name=name, documents_affected=document_count)
