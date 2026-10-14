import uuid

import pytest
from sqlalchemy import delete, func, select

from app.core.db import AsyncSessionLocal
from app.models import Document, DocumentSource, DocumentTag, Tag, TagAssignedBy, User
from app.models.user import AuthIdentity, AuthProvider
from app.services.tagging.rename import TagError, TagNotFound, rename_tag
from app.services.tagging.tags import get_or_create_tag


def t(name: str) -> str:
    """Namespace a tag name so the cleanup fixture can find it again."""
    return f"rn:{name}"


@pytest.fixture
async def rename_user(keep_rows):
    email = f"rename-{uuid.uuid4().hex[:8]}@example.com"
    async with AsyncSessionLocal() as db:
        user = User(email=email, email_verified=True)
        user.identities.append(
            AuthIdentity(
                provider=AuthProvider.PASSWORD,
                provider_user_id=email,
                password_hash="not-used-here",
            )
        )
        db.add(user)
        await db.commit()
        user_id = user.id

    yield user_id

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(User).where(User.id == user_id))
        await db.commit()


@pytest.fixture
async def make_document(rename_user, keep_rows):
    created: list[uuid.UUID] = []

    async def make():
        async with AsyncSessionLocal() as db:
            doc = Document(
                user_id=rename_user,
                source=DocumentSource.NOTE,
                content="a note in the rename tests",
                title="Rename note",
            )
            db.add(doc)
            await db.commit()
            created.append(doc.id)
            return doc.id

    yield make

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(created)))
        await db.commit()


@pytest.fixture(autouse=True)
async def clean_tags(keep_rows):
    yield
    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Tag).where(Tag.name.ilike("rn:%")))
        await db.commit()


async def mk_tag(db, name: str) -> Tag:
    tag = await get_or_create_tag(db, t(name))
    await db.flush()
    return tag


async def link(db, document_id, tag_id, assigned_by=TagAssignedBy.AUTO, confidence=None):
    db.add(
        DocumentTag(
            document_id=document_id,
            tag_id=tag_id,
            assigned_by=assigned_by,
            confidence=confidence,
        )
    )
    await db.flush()


async def links_for(tag_id) -> dict:
    async with AsyncSessionLocal() as db:
        rows = (
            (await db.execute(select(DocumentTag).where(DocumentTag.tag_id == tag_id)))
            .unique()
            .scalars()
            .all()
        )
        return {row.document_id: row for row in rows}


# a plain rename


@pytest.mark.anyio
async def test_rename_changes_the_name(make_document):
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "ML")
        await link(db, await make_document(), tag.id, TagAssignedBy.AUTO, 0.4)

        result = await rename_tag(db, tag.id, t("Machine Learning"))
        await db.commit()

    assert result.tag.name == t("Machine Learning")
    assert result.merged_into is None
    assert result.documents_affected == 1


@pytest.mark.anyio
async def test_rename_keeps_the_links_and_their_scores(make_document):
    """A rename is metadata. It must not disturb what the tag is on."""
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "ML")
        await link(db, document_id, tag.id, TagAssignedBy.AUTO, 0.4)
        tag_id = tag.id

        await rename_tag(db, tag_id, t("Machine Learning"))
        await db.commit()

    rows = await links_for(tag_id)
    assert list(rows) == [document_id]
    assert rows[document_id].confidence == pytest.approx(0.4)


@pytest.mark.anyio
async def test_a_case_only_change_is_not_a_merge():
    """
    Regression guard. lower(name) is unchanged, so the collision lookup
    finds THIS tag — and merging a tag into itself would delete the row it
    was meant to keep.
    """
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "machine learning")
        tag_id = tag.id

        result = await rename_tag(db, tag_id, t("Machine Learning"))
        await db.commit()

    assert result.tag.id == tag_id
    assert result.tag.name == t("Machine Learning")
    assert result.merged_into is None

    async with AsyncSessionLocal() as db:
        survivor = (await db.execute(select(Tag).where(Tag.id == tag_id))).scalar_one_or_none()
    assert survivor is not None


@pytest.mark.anyio
async def test_renaming_to_its_own_name_is_harmless():
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "Databases")
        result = await rename_tag(db, tag.id, f"  {t('Databases')}  ")
        await db.commit()

    assert result.tag.name == t("Databases")
    assert result.merged_into is None


@pytest.mark.anyio
async def test_the_new_name_is_normalized():
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "ML")
        result = await rename_tag(db, tag.id, f"  {t('machine')}   learning \n")
        await db.commit()

    assert result.tag.name == f"{t('machine')} learning"


# the merge


@pytest.mark.anyio
async def test_merge_moves_the_links_and_drops_the_source(make_document):
    first, second = await make_document(), await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, first, source_id, TagAssignedBy.AUTO, 0.5)
        await link(db, second, target_id, TagAssignedBy.AUTO, 0.6)

        result = await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    assert result.merged_into == target_id
    assert result.tag.id == target_id
    assert set(await links_for(target_id)) == {first, second}

    async with AsyncSessionLocal() as db:
        gone = (await db.execute(select(Tag).where(Tag.id == source_id))).scalar_one_or_none()
    assert gone is None


@pytest.mark.anyio
async def test_the_merge_target_is_matched_case_insensitively(make_document):
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, await make_document(), source_id)

        result = await rename_tag(db, source_id, t("MACHINE LEARNING"))
        await db.commit()

    assert result.merged_into == target_id


@pytest.mark.anyio
async def test_merge_does_not_duplicate_a_document_that_had_both(make_document):
    """uq / composite primary key would reject a second identical row."""
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, document_id, source_id, TagAssignedBy.AUTO, 0.5)
        await link(db, document_id, target_id, TagAssignedBy.AUTO, 0.6)

        await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    async with AsyncSessionLocal() as db:
        count = (
            await db.execute(
                select(func.count()).select_from(DocumentTag).where(DocumentTag.tag_id == target_id)
            )
        ).scalar_one()
    assert count == 1


@pytest.mark.anyio
async def test_merging_two_guesses_keeps_the_more_confident_one(make_document):
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, document_id, source_id, TagAssignedBy.AUTO, 0.9)
        await link(db, document_id, target_id, TagAssignedBy.AUTO, 0.3)

        await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    rows = await links_for(target_id)
    assert rows[document_id].confidence == pytest.approx(0.9)


@pytest.mark.anyio
async def test_merge_promotes_when_the_source_was_the_users_choice(make_document):
    """
    USER beats AUTO, and the promotion must clear the score —
    ck_document_tags_confidence_only_for_auto rejects the row otherwise.
    """
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, document_id, source_id, TagAssignedBy.USER, None)
        await link(db, document_id, target_id, TagAssignedBy.AUTO, 0.3)

        await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    rows = await links_for(target_id)
    assert rows[document_id].assigned_by is TagAssignedBy.USER
    assert rows[document_id].confidence is None


@pytest.mark.anyio
async def test_merge_never_demotes_the_users_choice(make_document):
    """
    The asymmetry that matters, same as in attach_tags: curating the
    vocabulary must not quietly turn a deliberate choice into a guess.
    """
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, document_id, source_id, TagAssignedBy.AUTO, 0.9)
        await link(db, document_id, target_id, TagAssignedBy.USER, None)

        await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    rows = await links_for(target_id)
    assert rows[document_id].assigned_by is TagAssignedBy.USER
    assert rows[document_id].confidence is None


@pytest.mark.anyio
async def test_a_user_link_with_no_overlap_arrives_as_user(make_document):
    """
    Without this, the insert could carry assigned_by wrongly and only the
    overlap case would notice.
    """
    document_id = await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id, target_id = source.id, target.id
        await link(db, document_id, source_id, TagAssignedBy.USER, None)

        await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    rows = await links_for(target_id)
    assert rows[document_id].assigned_by is TagAssignedBy.USER


@pytest.mark.anyio
async def test_documents_affected_is_measured_before_the_merge(make_document):
    """
    Counted after the merge it would read 0, because the source tag no
    longer exists — and the client would tell the user nothing changed.
    """
    a, b, c = await make_document(), await make_document(), await make_document()
    async with AsyncSessionLocal() as db:
        source = await mk_tag(db, "ML")
        target = await mk_tag(db, "Machine Learning")
        source_id = source.id
        await link(db, a, source_id)
        await link(db, b, source_id)
        await link(db, c, target.id)

        result = await rename_tag(db, source_id, t("Machine Learning"))
        await db.commit()

    assert result.documents_affected == 2


# validation


@pytest.mark.anyio
async def test_an_unknown_tag_id_is_not_found():
    async with AsyncSessionLocal() as db:
        with pytest.raises(TagNotFound):
            await rename_tag(db, uuid.uuid4(), t("Anything"))


@pytest.mark.anyio
async def test_a_blank_name_is_rejected():
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "ML")
        for blank in ("", "   ", "\n\t"):
            with pytest.raises(TagError, match="empty"):
                await rename_tag(db, tag.id, blank)


@pytest.mark.anyio
async def test_an_overlong_name_is_rejected_before_the_database():
    """String(64) would raise a DataError; this names the field and why."""
    async with AsyncSessionLocal() as db:
        tag = await mk_tag(db, "ML")
        with pytest.raises(TagError, match="limit"):
            await rename_tag(db, tag.id, "x" * 65)
