import uuid
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentTag, Tag, TagAssignedBy
from app.services.tagging.tags import TagError, normalize_tag_name


class TagNotFound(TagError):
    """No tag with that id. A subclass so callers may catch either."""


@dataclass(frozen=True)
class TagRenameResult:
    tag: Tag
    """The tag that SURVIVED. On a merge this is the absorbing tag, not the
    one whose id was passed in."""

    merged_into: uuid.UUID | None
    """Set when the requested tag was absorbed and no longer exists, so the
    client can say so instead of silently showing a different name."""

    documents_affected: int
    """How many documents visibly change, measured before anything moves:
    on a plain rename every document carrying the tag, on a merge every
    document that carried the source."""


async def _usage_count(db: AsyncSession, tag_id: uuid.UUID) -> int:
    return (
        await db.execute(
            select(func.count(DocumentTag.document_id)).where(DocumentTag.tag_id == tag_id)
        )
    ).scalar_one()


async def _by_lower_name(db: AsyncSession, name: str) -> Tag | None:
    return (
        await db.execute(select(Tag).where(func.lower(Tag.name) == name.lower()))
    ).scalar_one_or_none()


async def _merge(db: AsyncSession, *, source: Tag, target: Tag) -> None:
    src = DocumentTag.__table__.alias("src")

    # 1. Carry over every link the target does not already have. Overlaps
    #    are deliberately left alone here: the target's row wins for now,
    #    and steps 2 and 3 settle it. Doing it the other way round would
    #    let an AUTO source overwrite a USER target.
    await db.execute(
        pg_insert(DocumentTag)
        .from_select(
            ["document_id", "tag_id", "assigned_by", "confidence"],
            select(
                src.c.document_id,
                sa.literal(target.id),
                src.c.assigned_by,
                src.c.confidence,
            ).where(src.c.tag_id == source.id),
        )
        .on_conflict_do_nothing(index_elements=["document_id", "tag_id"])
    )

    # 2. The source said USER, so the survivor says USER. Clearing
    #    confidence is required, not cosmetic:
    #    ck_document_tags_confidence_only_for_auto rejects the row
    #    otherwise.
    await db.execute(
        update(DocumentTag)
        .where(
            DocumentTag.tag_id == target.id,
            exists(
                select(1).where(
                    src.c.tag_id == source.id,
                    src.c.document_id == DocumentTag.document_id,
                    src.c.assigned_by == TagAssignedBy.USER,
                )
            ),
        )
        .values(assigned_by=TagAssignedBy.USER, confidence=None)
    )

    # 3. Two guesses for one document: keep the more confident. GREATEST
    #    ignores NULLs in Postgres, so an AUTO link with no score loses to
    #    one that has a score rather than poisoning the result.
    await db.execute(
        update(DocumentTag)
        .where(
            DocumentTag.tag_id == target.id,
            DocumentTag.assigned_by == TagAssignedBy.AUTO,
            src.c.tag_id == source.id,
            src.c.document_id == DocumentTag.document_id,
            src.c.assigned_by == TagAssignedBy.AUTO,
        )
        .values(confidence=func.greatest(DocumentTag.confidence, src.c.confidence))
    )

    # 4. The source disappears. Its remaining links go with it through
    #    ON DELETE CASCADE on document_tags.tag_id.
    await db.execute(delete(Tag).where(Tag.id == source.id))

    # The row is gone, so the identity map must not keep handing it out or
    # try to flush it later.
    db.expunge(source)


async def rename_tag(db: AsyncSession, tag_id: uuid.UUID, new_name: str) -> TagRenameResult:
    """
    Rename a tag, merging it into an existing tag if the name is taken.

    Does not commit. The caller owns the transaction, as everywhere else
    in this package.
    """
    name = normalize_tag_name(new_name)

    tag = await db.get(Tag, tag_id)
    if tag is None:
        raise TagNotFound("tag not found")

    affected = await _usage_count(db, tag_id)

    # A case-only change must be handled before the collision lookup, or
    # that lookup finds THIS tag and the merge runs with source == target,
    # deleting the row it was supposed to keep. lower(name) is unchanged
    # here, so the unique index cannot object either way.
    if tag.name.lower() == name.lower():
        tag.name = name
        return TagRenameResult(tag=tag, merged_into=None, documents_affected=affected)

    target = await _by_lower_name(db, name)
    if target is not None:
        await _merge(db, source=tag, target=target)
        return TagRenameResult(tag=target, merged_into=target.id, documents_affected=affected)

    # The gap between the lookup above and this write is small but real:
    # two people curating the vocabulary at once can both find nothing and
    # both insert. The savepoint turns the resulting unique violation into
    # a 422 the client can retry, instead of poisoning the transaction.
    try:
        async with db.begin_nested():
            tag.name = name
            await db.flush()
    except IntegrityError as exc:
        raise TagError(f"the name {name!r} was taken by another request") from exc

    return TagRenameResult(tag=tag, merged_into=None, documents_affected=affected)
