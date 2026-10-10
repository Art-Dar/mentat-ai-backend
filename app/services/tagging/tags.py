"""
Tagging: attaching and removing tags on documents.

Tags are GLOBAL, not per-user — the same deliberate choice as documents,
where `user_id` is attribution rather than an access boundary. A shared
brain with two people's tag vocabularies kept apart would mean the same
"Machine Learning" existing twice and filtering returning half the results.

Two rules the database enforces and this module has to respect:

* `ix_tags_name_lower` is unique on `lower(name)`, so "Work" and "work"
  are the same tag. Every lookup goes through `lower()` or it silently
  misses and tries to insert a duplicate.

* `ck_document_tags_confidence_only_for_auto` forbids a confidence score
  on a user-assigned tag. A person either applied the tag or did not;
  there is no 0.7 about it.
"""

import logging
import re
import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import DocumentTag, Tag, TagAssignedBy

logger = logging.getLogger(__name__)

DEFAULT_TAG_COLOR = "#9CA3AF"
MAX_TAG_NAME_LENGTH = 64  # matches String(64) on tags.name

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_WHITESPACE = re.compile(r"\s+")


class TagError(ValueError):
    """A tag name or colour the database would reject."""


def normalize_tag_name(raw: str) -> str:
    """
    Trim and collapse whitespace, preserving the author's capitalisation.

    Display case is kept because "ML" and "Machine Learning" should look
    the way a person wrote them; uniqueness is case-insensitive separately,
    through the functional index. Lower-casing on the way in would make
    every tag in the sidebar shout in lowercase for no gain.
    """
    name = _WHITESPACE.sub(" ", raw).strip()
    if not name:
        raise TagError("tag name is empty")
    if len(name) > MAX_TAG_NAME_LENGTH:
        raise TagError(f"tag name is {len(name)} characters, limit is {MAX_TAG_NAME_LENGTH}")
    return name


def _validate_color(color: str) -> str:
    if not _HEX_COLOR.match(color):
        raise TagError(f"colour must be #RRGGBB, got {color!r}")
    return color


async def _find_by_name(db: AsyncSession, name: str) -> Tag | None:
    # func.lower() on both sides, matching ix_tags_name_lower exactly — an
    # equality test on Tag.name would miss "Work" when given "work" AND
    # would not use that index.
    result = await db.execute(select(Tag).where(func.lower(Tag.name) == name.lower()))
    return result.scalar_one_or_none()


async def get_or_create_tag(
    db: AsyncSession,
    name: str,
    *,
    color: str | None = None,
) -> Tag:
    """
    Fetch the tag with this name, creating it if it does not exist.

    An existing tag keeps its colour: `color` is a suggestion for a new
    tag, not an update. Auto-tagging runs on every ingest, and a tag the
    user recoloured must not revert the next time Claude mentions it.
    """
    name = normalize_tag_name(name)
    color = _validate_color(color) if color else DEFAULT_TAG_COLOR

    existing = await _find_by_name(db, name)
    if existing is not None:
        return existing

    tag = Tag(name=name, color=color)
    try:
        # SAVEPOINT, so losing the race rolls back only this INSERT rather
        # than the caller's whole transaction. Two background tasks
        # ingesting at once will both try to create "Machine Learning".
        async with db.begin_nested():
            db.add(tag)
            await db.flush()
    except IntegrityError:
        logger.debug("lost the race creating tag %r, reusing the winner", name)
        winner = await _find_by_name(db, name)
        if winner is None:  # the unique violation came from something else
            raise
        return winner

    return tag


async def attach_tags(
    db: AsyncSession,
    document_id: uuid.UUID,
    names: Sequence[str],
    *,
    assigned_by: TagAssignedBy = TagAssignedBy.AUTO,
    confidence: float | None = None,
) -> list[Tag]:
    """
    Attach tags to a document, creating any that do not exist yet.

    Idempotent, and asymmetric on purpose:

      * attaching as USER over an existing AUTO link promotes it — the
        person confirmed the guess — and clears the confidence score,
        which the CHECK constraint requires.
      * attaching as AUTO over an existing link does nothing. A later
        auto-tagging pass must never demote a tag the user applied.
    """
    if assigned_by is TagAssignedBy.USER and confidence is not None:
        raise TagError(
            "a user-assigned tag cannot carry a confidence score "
            "(ck_document_tags_confidence_only_for_auto)"
        )
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise TagError(f"confidence must be between 0 and 1, got {confidence}")

    tags = [await get_or_create_tag(db, name) for name in names]
    if not tags:
        return []

    rows = [
        {
            "document_id": document_id,
            "tag_id": tag.id,
            "assigned_by": assigned_by,
            "confidence": confidence,
        }
        for tag in tags
    ]
    statement = insert(DocumentTag).values(rows)

    if assigned_by is TagAssignedBy.USER:
        statement = statement.on_conflict_do_update(
            index_elements=["document_id", "tag_id"],
            set_={"assigned_by": TagAssignedBy.USER.value, "confidence": None},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=["document_id", "tag_id"])

    await db.execute(statement)
    return tags


async def detach_tag(db: AsyncSession, document_id: uuid.UUID, name: str) -> bool:
    """
    Remove one tag from one document. Returns whether a link was removed.

    The Tag row itself survives — it is probably on other documents, and a
    tag disappearing from the sidebar because the last document using it
    was untagged would be surprising. Orphaned tags are cheap; deleting
    shared rows as a side effect is not.
    """
    tag = await _find_by_name(db, normalize_tag_name(name))
    if tag is None:
        return False

    result = await db.execute(
        delete(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.tag_id == tag.id,
        )
    )
    return result.rowcount > 0


async def replace_auto_tags(
    db: AsyncSession,
    document_id: uuid.UUID,
    names: Sequence[str],
    *,
    confidence: float | None = None,
) -> list[Tag]:
    """
    Swap this document's automatic tags for a new set, leaving the user's
    tags alone.

    This is what a re-run of auto-tagging calls. Deleting every link and
    re-adding would silently discard the tags a person applied by hand,
    which is the kind of data loss nobody reports because they assume they
    forgot to save.
    """
    await db.execute(
        delete(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.assigned_by == TagAssignedBy.AUTO,
        )
    )
    return await attach_tags(
        db,
        document_id,
        names,
        assigned_by=TagAssignedBy.AUTO,
        confidence=confidence,
    )


async def tags_for_document(db: AsyncSession, document_id: uuid.UUID) -> list[DocumentTag]:
    """
    This document's tag links, each with its Tag loaded.

    selectinload, so rendering a list of tags is two queries rather than
    one per tag — and so touching `.tag` outside the session does not
    raise MissingGreenlet, which is how lazy loading fails under asyncio.
    """
    result = await db.execute(
        select(DocumentTag)
        .where(DocumentTag.document_id == document_id)
        .options(selectinload(DocumentTag.tag))
        .join(Tag, Tag.id == DocumentTag.tag_id)
        .order_by(func.lower(Tag.name))
    )
    return list(result.scalars().all())


async def documents_for_tag(db: AsyncSession, name: str) -> list[uuid.UUID]:
    """Every document carrying this tag. The extension's filter view."""
    tag = await _find_by_name(db, normalize_tag_name(name))
    if tag is None:
        return []

    result = await db.execute(select(DocumentTag.document_id).where(DocumentTag.tag_id == tag.id))
    return list(result.scalars().all())


async def list_tags(db: AsyncSession) -> list[tuple[Tag, int]]:
    """
    Every tag with how many documents use it, most used first.

    An outer join so a tag with no documents still appears with a count of
    zero, rather than vanishing from the sidebar the moment it is unused.
    """
    result = await db.execute(
        select(Tag, func.count(DocumentTag.document_id))
        .outerjoin(DocumentTag, DocumentTag.tag_id == Tag.id)
        .group_by(Tag.id)
        .order_by(func.count(DocumentTag.document_id).desc(), func.lower(Tag.name))
    )
    return [(tag, count) for tag, count in result.all()]
