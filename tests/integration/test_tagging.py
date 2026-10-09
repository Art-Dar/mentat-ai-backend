import uuid

import pytest
from sqlalchemy import delete, func, select

from app.core.db import AsyncSessionLocal
from app.models import Document, DocumentSource, DocumentTag, Tag, TagAssignedBy, User
from app.models.user import AuthIdentity, AuthProvider
from app.services.tagging.tags import (
    DEFAULT_TAG_COLOR,
    MAX_TAG_NAME_LENGTH,
    TagError,
    attach_tags,
    detach_tag,
    documents_for_tag,
    get_or_create_tag,
    list_tags,
    normalize_tag_name,
    replace_auto_tags,
    tags_for_document,
)


@pytest.fixture
async def tag_user(keep_rows):
    email = f"tags-{uuid.uuid4().hex[:8]}@example.com"
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
async def document(tag_user, keep_rows):
    """A document to hang tags on. Content is irrelevant here."""
    async with AsyncSessionLocal() as db:
        doc = Document(
            user_id=tag_user,
            source=DocumentSource.NOTE,
            content="a note that needs tagging",
            title="Tagged note",
        )
        db.add(doc)
        await db.commit()
        document_id = doc.id

    yield document_id

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id == document_id))
        await db.commit()


@pytest.fixture(autouse=True)
async def clean_tags(keep_rows):
    """
    Tags are global, so they outlive the document fixtures that created
    them and would leak between tests. Named tags used here are prefixed
    so this only removes its own.
    """
    yield
    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Tag).where(Tag.name.ilike("t:%")))
        await db.commit()


def t(name: str) -> str:
    """Namespace a tag name so the cleanup fixture can find it again."""
    return f"t:{name}"


async def links(document_id: uuid.UUID) -> list[DocumentTag]:
    async with AsyncSessionLocal() as db:
        return await tags_for_document(db, document_id)


# names


@pytest.mark.anyio
async def test_name_whitespace_is_collapsed_and_trimmed():
    assert normalize_tag_name("  machine   learning \n") == "machine learning"


@pytest.mark.anyio
async def test_blank_name_is_rejected():
    for blank in ("", "   ", "\n\t"):
        with pytest.raises(TagError, match="empty"):
            normalize_tag_name(blank)


@pytest.mark.anyio
async def test_overlong_name_is_rejected_before_the_database():
    """String(64) would raise a DataError; this says which field and why."""
    with pytest.raises(TagError, match="limit"):
        normalize_tag_name("x" * (MAX_TAG_NAME_LENGTH + 1))


# get_or_create


@pytest.mark.anyio
async def test_tag_is_created_with_the_default_colour():
    async with AsyncSessionLocal() as db:
        tag = await get_or_create_tag(db, t("Databases"))
        await db.commit()

    assert tag.name == t("Databases")
    assert tag.color == DEFAULT_TAG_COLOR


@pytest.mark.anyio
async def test_lookup_is_case_insensitive():
    """ix_tags_name_lower is unique on lower(name): one tag, not three."""
    async with AsyncSessionLocal() as db:
        first = await get_or_create_tag(db, t("Work"))
        second = await get_or_create_tag(db, t("work"))
        third = await get_or_create_tag(db, f"  {t('WORK')}  ")
        await db.commit()

    assert first.id == second.id == third.id

    async with AsyncSessionLocal() as db:
        count = (
            await db.execute(select(func.count(Tag.id)).where(func.lower(Tag.name) == t("work")))
        ).scalar_one()
    assert count == 1


@pytest.mark.anyio
async def test_original_capitalisation_is_kept():
    async with AsyncSessionLocal() as db:
        tag = await get_or_create_tag(db, t("Machine Learning"))
        await db.commit()

    assert tag.name == t("Machine Learning")


@pytest.mark.anyio
async def test_an_existing_tag_keeps_its_colour():
    """
    Auto-tagging runs on every ingest. A colour the user chose must not be
    reverted the next time Claude happens to mention the same tag.
    """
    async with AsyncSessionLocal() as db:
        await get_or_create_tag(db, t("Recipes"), color="#FF0000")
        await db.commit()

    async with AsyncSessionLocal() as db:
        again = await get_or_create_tag(db, t("recipes"), color="#00FF00")
        await db.commit()

    assert again.color == "#FF0000"


@pytest.mark.anyio
async def test_bad_colour_is_rejected():
    async with AsyncSessionLocal() as db:
        for bad in ("red", "#FFF", "#GGGGGG", "FF0000"):
            with pytest.raises(TagError, match="RRGGBB"):
                await get_or_create_tag(db, t("x"), color=bad)


# attach


@pytest.mark.anyio
async def test_attaching_creates_links(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python"), t("fastapi")])
        await db.commit()

    names = {link.tag.name for link in await links(document)}
    assert names == {t("python"), t("fastapi")}


@pytest.mark.anyio
async def test_attaching_is_idempotent(document):
    """The composite primary key would reject a second identical row."""
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python")])
        await attach_tags(db, document, [t("python"), t("PYTHON")])
        await db.commit()

    assert len(await links(document)) == 1


@pytest.mark.anyio
async def test_auto_tags_store_their_confidence(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("ml")], confidence=0.82)
        await db.commit()

    link = (await links(document))[0]
    assert link.assigned_by is TagAssignedBy.AUTO
    assert link.confidence == pytest.approx(0.82)


@pytest.mark.anyio
async def test_user_tags_cannot_carry_confidence(document):
    """
    ck_document_tags_confidence_only_for_auto. Caught in Python so the
    error names the rule instead of surfacing as an IntegrityError.
    """
    async with AsyncSessionLocal() as db:
        with pytest.raises(TagError, match="confidence"):
            await attach_tags(
                db,
                document,
                [t("python")],
                assigned_by=TagAssignedBy.USER,
                confidence=0.9,
            )


@pytest.mark.anyio
async def test_confidence_outside_zero_to_one_is_rejected(document):
    async with AsyncSessionLocal() as db:
        for bad in (-0.1, 1.5):
            with pytest.raises(TagError, match="between 0 and 1"):
                await attach_tags(db, document, [t("python")], confidence=bad)


@pytest.mark.anyio
async def test_a_user_attach_promotes_an_auto_tag(document):
    """Confirming a guess makes it the user's, and clears the score."""
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python")], confidence=0.6)
        await db.commit()

    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python")], assigned_by=TagAssignedBy.USER)
        await db.commit()

    link = (await links(document))[0]
    assert link.assigned_by is TagAssignedBy.USER
    assert link.confidence is None


@pytest.mark.anyio
async def test_an_auto_pass_never_demotes_a_user_tag(document):
    """
    The asymmetry that matters: re-running auto-tagging must not quietly
    turn a deliberate choice back into a guess.
    """
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python")], assigned_by=TagAssignedBy.USER)
        await db.commit()

    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python")], confidence=0.4)
        await db.commit()

    link = (await links(document))[0]
    assert link.assigned_by is TagAssignedBy.USER
    assert link.confidence is None


@pytest.mark.anyio
async def test_attaching_nothing_is_not_an_error(document):
    async with AsyncSessionLocal() as db:
        assert await attach_tags(db, document, []) == []
        await db.commit()

    assert await links(document) == []


# detach


@pytest.mark.anyio
async def test_detaching_removes_the_link_but_keeps_the_tag(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("python"), t("fastapi")])
        await db.commit()

    async with AsyncSessionLocal() as db:
        assert await detach_tag(db, document, t("PYTHON")) is True
        await db.commit()

    assert {link.tag.name for link in await links(document)} == {t("fastapi")}

    async with AsyncSessionLocal() as db:
        survivor = (
            await db.execute(select(Tag).where(func.lower(Tag.name) == t("python")))
        ).scalar_one_or_none()
    assert survivor is not None


@pytest.mark.anyio
async def test_detaching_an_unknown_tag_returns_false(document):
    async with AsyncSessionLocal() as db:
        assert await detach_tag(db, document, t("never-used")) is False


@pytest.mark.anyio
async def test_detaching_a_tag_the_document_does_not_have_returns_false(document):
    async with AsyncSessionLocal() as db:
        await get_or_create_tag(db, t("elsewhere"))
        await db.commit()

    async with AsyncSessionLocal() as db:
        assert await detach_tag(db, document, t("elsewhere")) is False


# replace_auto_tags


@pytest.mark.anyio
async def test_replacing_auto_tags_keeps_the_users_own(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("mine")], assigned_by=TagAssignedBy.USER)
        await attach_tags(db, document, [t("guess-one"), t("guess-two")], confidence=0.5)
        await db.commit()

    async with AsyncSessionLocal() as db:
        await replace_auto_tags(db, document, [t("guess-three")], confidence=0.7)
        await db.commit()

    by_name = {link.tag.name: link for link in await links(document)}
    assert set(by_name) == {t("mine"), t("guess-three")}
    assert by_name[t("mine")].assigned_by is TagAssignedBy.USER
    assert by_name[t("guess-three")].confidence == pytest.approx(0.7)


@pytest.mark.anyio
async def test_replacing_with_nothing_clears_only_auto_tags(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("mine")], assigned_by=TagAssignedBy.USER)
        await attach_tags(db, document, [t("guess")], confidence=0.5)
        await db.commit()

    async with AsyncSessionLocal() as db:
        await replace_auto_tags(db, document, [])
        await db.commit()

    assert {link.tag.name for link in await links(document)} == {t("mine")}


# reads

@pytest.mark.anyio
async def test_tags_for_a_document_come_back_sorted(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("zebra"), t("Apple"), t("mango")])
        await db.commit()

    assert [link.tag.name for link in await links(document)] == [
        t("Apple"),
        t("mango"),
        t("zebra"),
    ]


@pytest.mark.anyio
async def test_documents_for_a_tag(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("shared")])
        await db.commit()

    async with AsyncSessionLocal() as db:
        assert await documents_for_tag(db, t("SHARED")) == [document]
        assert await documents_for_tag(db, t("nonexistent")) == []


@pytest.mark.anyio
async def test_list_tags_counts_usage_and_keeps_unused_ones(document):
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("used")])
        await get_or_create_tag(db, t("unused"))
        await db.commit()

    async with AsyncSessionLocal() as db:
        counts = {tag.name: n for tag, n in await list_tags(db) if tag.name.startswith("t:")}

    assert counts[t("used")] == 1
    assert counts[t("unused")] == 0


# cascades


@pytest.mark.anyio
async def test_deleting_a_document_removes_its_links(tag_user, keep_rows):
    """
    ON DELETE CASCADE on document_tags.document_id. Without it, deleting a
    document leaves rows pointing at nothing and every join on tags starts
    returning ghosts.
    """
    async with AsyncSessionLocal() as db:
        doc = Document(user_id=tag_user, source=DocumentSource.NOTE, content="temporary")
        db.add(doc)
        await db.commit()
        doc_id = doc.id
        await attach_tags(db, doc_id, [t("doomed")])
        await db.commit()

    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id == doc_id))
        await db.commit()

    async with AsyncSessionLocal() as db:
        remaining = (
            await db.execute(
                select(func.count())
                .select_from(DocumentTag)
                .where(DocumentTag.document_id == doc_id)
            )
        ).scalar_one()
    assert remaining == 0


@pytest.mark.anyio
async def test_deleting_a_tag_removes_its_links(document):
    """ON DELETE CASCADE on document_tags.tag_id."""
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document, [t("temporary")])
        await db.commit()

    async with AsyncSessionLocal() as db:
        await db.execute(delete(Tag).where(func.lower(Tag.name) == t("temporary")))
        await db.commit()

    assert await links(document) == []