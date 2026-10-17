import uuid

import pytest
from sqlalchemy import delete

from app.core.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.user import AuthIdentity, AuthProvider
from app.services.answering.answerer import answer_question
from app.services.answering.answering_prompt import MIN_USEFUL_QUOTE_CHARS, Turn
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.llm.fake_client import FakeChatClient

# Under MIN_USEFUL_QUOTE_CHARS, so the prompt drops it — and it is the
# text the query matches exactly, so it also ranks FIRST. Dropping the
# top-ranked source is what forces the renumbering this file is about.
STUB = "Total 480 UAH."

LONG = (
    "Payment terms are net 14 days from the invoice date, with a two "
    "percent discount for settlement within five working days of receipt."
)


class RecordingEmbedder(FakeEmbedder):
    """FakeEmbedder, but it remembers what text it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


@pytest.fixture
async def owner(keep_rows):
    email = f"ans-{uuid.uuid4().hex[:8]}@example.com"
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
async def two_documents(owner, keep_rows):
    """
    One document whose only chunk is too short to be worth quoting, and
    one with a real passage. Separate documents so the citations are
    distinguishable by title.
    """
    ids = []
    async with AsyncSessionLocal() as db:
        for title, content in (("Sticky note", STUB), ("Invoice 2026-0042", LONG)):
            document = Document(
                user_id=owner,
                source=DocumentSource.NOTE if title == "Sticky note" else DocumentSource.PDF,
                status=IngestionStatus.COMPLETED,
                content=content,
                title=title,
            )
            db.add(document)
            await db.flush()
            db.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=0,
                    content=content,
                    start_char=0,
                    end_char=len(content),
                    embedding=(await FakeEmbedder().embed_documents([content]))[0],
                    embedding_model="fake",
                )
            )
            ids.append(document.id)
        await db.commit()

    yield ids

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(ids)))
        await db.commit()


@pytest.mark.anyio
async def test_the_stub_really_is_dropped_and_really_does_rank_first(two_documents):
    """
    The premise of the test below, asserted separately so that a change
    making the setup meaningless fails loudly instead of quietly turning
    the next test into one that proves nothing.
    """
    assert len(" ".join(STUB.split())) < MIN_USEFUL_QUOTE_CHARS

    async with AsyncSessionLocal() as db:
        from app.services.retrieval.search import search_by_text

        hits = await search_by_text(
            db,
            STUB,
            user_id=(await db.get(Document, two_documents[0])).user_id,
            embedder=FakeEmbedder(),
            top_k=5,
            min_similarity=-1.0,
        )

    assert hits[0].content == STUB, "the short chunk should rank first"


@pytest.mark.anyio
async def test_markers_resolve_against_what_the_model_was_shown(owner, two_documents):
    """
    The renumbering seam.

    The short chunk ranks first and is then dropped from the prompt for
    being too short, so the one source the model sees is the invoice,
    renumbered [1]. If grounding runs against the pre-trim list, [1]
    resolves to the sticky note — the answer would be about the invoice
    and the footnote would point at a different document.
    """
    async with AsyncSessionLocal() as db:
        answer = await answer_question(
            db,
            question=STUB,
            user_id=owner,
            embedder=FakeEmbedder(),
            chat=FakeChatClient("Net 14 days from the invoice date [1]."),
            top_k=5,
            min_similarity=-1.0,
        )

    assert [c.marker for c in answer.cited] == ["[1]"]
    assert answer.cited[0].title == "Invoice 2026-0042", (
        "[1] resolved to the source that was dropped from the prompt"
    )
    assert answer.dropped_sources == 1


@pytest.mark.anyio
async def test_the_prompt_only_contains_the_sources_that_survived(owner, two_documents):
    recorder = FakeChatClient("Net 14 days [1].")

    async with AsyncSessionLocal() as db:
        await answer_question(
            db,
            question=STUB,
            user_id=owner,
            embedder=FakeEmbedder(),
            chat=recorder,
            top_k=5,
            min_similarity=-1.0,
        )

    prompt = recorder.calls[0]["user"]
    assert "<<<SOURCE [1]>>>" in prompt
    assert "<<<SOURCE [2]>>>" not in prompt, "a dropped source still reached the model"


@pytest.mark.anyio
async def test_the_condensed_question_is_what_gets_embedded(owner, two_documents):
    """
    The condensation is worthless if the raw follow-up is what reaches
    the embedder. The response reports `search_query` either way, so
    only the embedder can say which text was actually searched for.
    """
    embedder = RecordingEmbedder()
    history = (
        Turn(role="user", content="what did I save about coffee?"),
        Turn(role="assistant", content="An invoice from Kyiv Coffee Roasters [1]."),
    )

    async with AsyncSessionLocal() as db:
        answer = await answer_question(
            db,
            question="when is it due?",
            user_id=owner,
            embedder=embedder,
            chat=FakeChatClient(
                ["when is the Kyiv Coffee Roasters invoice due?", "Net 14 days [1]."]
            ),
            history=history,
            top_k=5,
            min_similarity=-1.0,
        )

    assert embedder.queries == ["when is the Kyiv Coffee Roasters invoice due?"]
    assert answer.search_query == "when is the Kyiv Coffee Roasters invoice due?"
    assert answer.question == "when is it due?"


@pytest.mark.anyio
async def test_the_raw_question_is_embedded_when_there_is_no_history(owner, two_documents):
    embedder = RecordingEmbedder()

    async with AsyncSessionLocal() as db:
        await answer_question(
            db,
            question="when is the invoice due?",
            user_id=owner,
            embedder=embedder,
            chat=FakeChatClient("Net 14 days [1]."),
            top_k=5,
            min_similarity=-1.0,
        )

    assert embedder.queries == ["when is the invoice due?"]


@pytest.mark.anyio
async def test_history_reaches_the_prompt_oldest_first(owner, two_documents):
    """
    Reversed history reads as the assistant speaking before it was
    asked, and the model resolves "it" against the wrong turn.
    """
    recorder = FakeChatClient(["standalone question", "Net 14 days [1]."])
    history = (
        Turn(role="user", content="FIRST QUESTION about coffee"),
        Turn(role="assistant", content="SECOND the reply about an invoice"),
        Turn(role="user", content="THIRD a follow up"),
    )

    async with AsyncSessionLocal() as db:
        await answer_question(
            db,
            question="and the total?",
            user_id=owner,
            embedder=FakeEmbedder(),
            chat=recorder,
            history=history,
            top_k=5,
            min_similarity=-1.0,
        )

    prompt = recorder.calls[1]["user"]
    assert prompt.index("FIRST") < prompt.index("SECOND") < prompt.index("THIRD")
    assert "User: FIRST QUESTION about coffee" in prompt
    assert "Assistant: SECOND the reply about an invoice" in prompt
