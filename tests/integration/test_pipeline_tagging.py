"""
Auto-tagging inside the real pipeline, against a real database.
"""

import uuid

import pytest
from pydantic import SecretStr
from sqlalchemy import delete, func, select

from app.core.db import AsyncSessionLocal
from app.models import (
    Chunk,
    Document,
    DocumentSource,
    DocumentTag,
    IngestionStatus,
    Tag,
    TagAssignedBy,
    User,
)
from app.models.user import AuthIdentity, AuthProvider
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.ingestion.pipeline import process_document
from app.services.tagging.auto_tagger import TaggingError, TagSuggestion
from app.services.tagging.fake_tagger import FakeTagger
from app.services.tagging.tags import attach_tags, tags_for_document

ARTICLE = "\n\n".join(
    [
        "Retrieval augmented generation combines a vector search over your own "
        "documents with a language model that answers from what it retrieves. "
        "The quality of the answer depends far more on retrieval than on the "
        "model, which is why chunking and embedding choices matter so much.",
        "Нормалізація тексту виконується перед розбиттям на фрагменти. Вона "
        "приводить пробіли до єдиного вигляду та застосовує форму NFKC до "
        "символів Юнікоду, інакше однакові за змістом тексти дають різні "
        "вектори і дублікати неможливо виявити за хешем вмісту.",
    ]
)


class BrokenTagger:
    """Groq unavailable after the retries are exhausted."""

    async def suggest_tags(self, *, title, text, vocabulary):
        raise TaggingError("Claude unavailable after 3 attempts")


class FixedTagger:
    """Returns exactly these suggestions, so assertions can name them."""

    def __init__(self, *names_and_scores):
        self.suggestions = [TagSuggestion(n, c) for n, c in names_and_scores]
        self.vocabularies: list[list[str]] = []

    async def suggest_tags(self, *, title, text, vocabulary):
        self.vocabularies.append(list(vocabulary))
        return self.suggestions


@pytest.fixture
async def tagging_user(keep_rows):
    email = f"autotag-{uuid.uuid4().hex[:8]}@example.com"
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
async def captured(tagging_user, keep_rows):
    created: list[uuid.UUID] = []

    async def make(text: str = ARTICLE, title: str = "RAG and normalization"):
        async with AsyncSessionLocal() as db:
            document = Document(
                user_id=tagging_user,
                source=DocumentSource.ARTICLE,
                content=text,
                title=title,
            )
            db.add(document)
            await db.commit()
            created.append(document.id)
            return document.id

    yield make

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(created)))
        await db.execute(delete(Tag).where(Tag.name.ilike("at:%")))
        await db.commit()


async def links(document_id: uuid.UUID):
    async with AsyncSessionLocal() as db:
        return await tags_for_document(db, document_id)


async def chunk_count(document_id: uuid.UUID) -> int:
    async with AsyncSessionLocal() as db:
        return (
            await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document_id))
        ).scalar_one()


# the happy path


@pytest.mark.anyio
async def test_suggested_tags_are_stored(captured):
    document_id = await captured()
    tagger = FixedTagger(("at:Retrieval", 0.93), ("at:Embeddings", 0.71))

    await process_document(document_id, embedder=FakeEmbedder(), tagger=tagger)

    by_name = {link.tag.name: link for link in await links(document_id)}
    assert set(by_name) == {"at:Retrieval", "at:Embeddings"}
    assert by_name["at:Retrieval"].confidence == pytest.approx(0.93)


@pytest.mark.anyio
async def test_tags_are_marked_as_automatic(captured):
    """So the UI can show a guess differently from the user's own choice."""
    document_id = await captured()

    await process_document(
        document_id, embedder=FakeEmbedder(), tagger=FixedTagger(("at:Guess", 0.5))
    )

    assert (await links(document_id))[0].assigned_by is TagAssignedBy.AUTO


@pytest.mark.anyio
async def test_the_existing_vocabulary_is_offered_to_the_tagger(captured):
    """
    The anti-fragmentation mechanism, end to end: a tag already in the
    database must reach the model as a reuse candidate.
    """
    async with AsyncSessionLocal() as db:
        await attach_tags(db, await captured(), ["at:Machine Learning"])
        await db.commit()

    document_id = await captured()
    tagger = FixedTagger(("at:Machine Learning", 0.9))

    await process_document(document_id, embedder=FakeEmbedder(), tagger=tagger)

    assert "at:Machine Learning" in tagger.vocabularies[0]


@pytest.mark.anyio
async def test_the_document_title_reaches_the_tagger(captured):
    document_id = await captured(title="at:Specific Title")
    seen: list[str | None] = []

    class TitleSpy:
        async def suggest_tags(self, *, title, text, vocabulary):
            seen.append(title)
            return []

    await process_document(document_id, embedder=FakeEmbedder(), tagger=TitleSpy())

    assert seen == ["at:Specific Title"]


@pytest.mark.anyio
async def test_the_tagger_sees_normalized_text_not_the_raw_capture(captured):
    """Tagging runs after normalization, so entities and spacing are clean."""
    document_id = await captured(text=ARTICLE.replace(" ", "&nbsp;  "))
    seen: list[str] = []

    class TextSpy:
        async def suggest_tags(self, *, title, text, vocabulary):
            seen.append(text)
            return []

    await process_document(document_id, embedder=FakeEmbedder(), tagger=TextSpy())

    assert "&nbsp;" not in seen[0]


# the failure policy


@pytest.mark.anyio
async def test_a_tagging_failure_does_not_fail_the_document(captured):
    """
    The whole reason tagging has its own except clause. Claude being
    overloaded must not hide a perfectly good document from the user.
    """
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder(), tagger=BrokenTagger())

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
    assert document.status is IngestionStatus.COMPLETED
    assert document.error_message is None


@pytest.mark.anyio
async def test_a_tagging_failure_leaves_the_embeddings_intact(captured):
    """
    The savepoint, not just the except clause. Without it the failed
    tagging transaction would poison the session and take the chunks down
    with it at the outer commit.
    """
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder(), tagger=BrokenTagger())

    assert await chunk_count(document_id) > 0
    assert await links(document_id) == []


@pytest.mark.real_providers
@pytest.mark.anyio
async def test_no_api_key_costs_the_tags_not_the_document(captured, monkeypatch):
    """
    groq_api_key is optional, unlike voyage_api_key. Passing no tagger
    makes the pipeline build the real one, which refuses without a key —
    and that refusal is absorbed.
    """
    # Patch target follows the key check into the shared LLM client. What
    # is being pinned is unchanged: no key costs the tags, never the
    # document.
    monkeypatch.setattr("app.services.llm.groq_client.settings.groq_api_key", SecretStr(""))
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
    assert document.status is IngestionStatus.COMPLETED
    assert await chunk_count(document_id) > 0


# re-running


@pytest.mark.anyio
async def test_reprocessing_replaces_the_previous_guesses(captured):
    document_id = await captured()
    await process_document(
        document_id, embedder=FakeEmbedder(), tagger=FixedTagger(("at:Old", 0.5))
    )

    await process_document(
        document_id, embedder=FakeEmbedder(), tagger=FixedTagger(("at:New", 0.8))
    )

    assert {link.tag.name for link in await links(document_id)} == {"at:New"}


@pytest.mark.anyio
async def test_reprocessing_keeps_the_tags_the_user_applied(captured):
    """
    The data loss nobody reports, because they assume they forgot to save:
    a re-run of auto-tagging silently discarding hand-applied tags.
    """
    document_id = await captured()
    await process_document(
        document_id, embedder=FakeEmbedder(), tagger=FixedTagger(("at:Guess", 0.5))
    )
    async with AsyncSessionLocal() as db:
        await attach_tags(db, document_id, ["at:Mine"], assigned_by=TagAssignedBy.USER)
        await db.commit()

    await process_document(
        document_id, embedder=FakeEmbedder(), tagger=FixedTagger(("at:Other", 0.6))
    )

    by_name = {link.tag.name: link for link in await links(document_id)}
    assert set(by_name) == {"at:Mine", "at:Other"}
    assert by_name["at:Mine"].assigned_by is TagAssignedBy.USER


@pytest.mark.anyio
async def test_a_tag_shared_by_two_documents_is_one_row(captured):
    """Tags are global: the vocabulary converges rather than duplicating."""
    first = await captured()
    second = await captured()
    tagger = FixedTagger(("at:Shared", 0.9))

    await process_document(first, embedder=FakeEmbedder(), tagger=tagger)
    await process_document(second, embedder=FakeEmbedder(), tagger=tagger)

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(select(func.count(Tag.id)).where(Tag.name == "at:Shared"))
        ).scalar_one()
        usages = (
            await db.execute(
                select(func.count(DocumentTag.document_id))
                .join(Tag, Tag.id == DocumentTag.tag_id)
                .where(Tag.name == "at:Shared", DocumentTag.document_id.in_([first, second]))
            )
        ).scalar_one()

    assert rows == 1
    assert usages == 2


@pytest.mark.anyio
async def test_the_offline_fake_tagger_works_through_the_pipeline(captured):
    """Lets the whole pipeline run with no API keys of any kind."""
    document_id = await captured()
    fake = FakeTagger()

    await process_document(document_id, embedder=FakeEmbedder(), tagger=fake)

    assert fake.calls  # it was asked
    assert await links(document_id)  # and something was stored
