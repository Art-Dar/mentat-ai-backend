#The ingestion pipeline against a real database.
#integration tests

import uuid

import pytest
from sqlalchemy import delete, func, select

from app.core.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.models.user import AuthIdentity, AuthProvider
from app.services.ingestion.embedder import EmbeddingError
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.ingestion.pipeline import process_document

# Long enough to produce several chunks at DEFAULT_CHUNK_SIZE = 1000.
PARAGRAPHS = [
    "Vector databases store high-dimensional embeddings and retrieve them by "
    "approximate nearest neighbour search. The HNSW index builds a navigable "
    "small-world graph over the vectors, so recall stays high without scanning "
    "every row in the table.",
    "Chunking strategy matters more than most teams expect. If a chunk is too "
    "large, one relevant sentence is diluted by a thousand characters of "
    "unrelated prose. If it is too small, it loses the context needed to make "
    "sense on its own when it comes back as a citation.",
    "Нормалізація тексту виконується перед розбиттям на фрагменти. Вона "
    "приводить пробіли до єдиного вигляду, розкодовує HTML-сутності та "
    "застосовує форму NFKC до символів Юнікоду. Без цього однакові за змістом "
    "тексти дають різні вектори.",
    "Citations require exact character offsets into the stored document. The "
    "chunker records start_char and end_char for every chunk, and a chunker "
    "that rewrites text while splitting breaks those offsets silently.",
]
ARTICLE = "\n\n".join(PARAGRAPHS)


class BrokenEmbedder:
    """Stands in for Voyage being down after the retries are exhausted."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Voyage unavailable after 3 attempts")

    async def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("Voyage unavailable after 3 attempts")


@pytest.fixture
async def pipeline_user(keep_rows):
    """A user to own the test documents, removed afterwards."""
    email = f"pipeline-{uuid.uuid4().hex[:8]}@example.com"
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
        print(f"\n--keep-rows: user {user_id} left in the database")
        return

    async with AsyncSessionLocal() as db:
        await db.execute(delete(User).where(User.id == user_id))
        await db.commit()


@pytest.fixture
async def captured(pipeline_user, keep_rows):
    """
    Factory for documents in the state /ingest leaves them: PENDING, raw
    text in `content`, nothing else done.

    Every document it creates is deleted at the end of the test. Chunks go
    with them through ON DELETE CASCADE — which these tests therefore also
    exercise, every single run.
    """
    created: list[uuid.UUID] = []

    async def make(text: str = ARTICLE, source: DocumentSource = DocumentSource.ARTICLE):
        async with AsyncSessionLocal() as db:
            document = Document(
                user_id=pipeline_user,
                source=source,
                content=text,
                url="https://example.com/test",
                title="Test capture",
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
        await db.commit()


async def load(document_id: uuid.UUID) -> tuple[Document, list[Chunk]]:
    """Read a document and its chunks back in a fresh session, in order."""
    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        chunks = (
            (
                await db.execute(
                    select(Chunk)
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.chunk_index)
                )
            )
            .scalars()
            .all()
        )
        return document, list(chunks)


# the happy path


@pytest.mark.anyio
async def test_document_reaches_completed(captured):
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    document, chunks = await load(document_id)
    assert document.status is IngestionStatus.COMPLETED
    assert document.error_message is None
    assert chunks


@pytest.mark.anyio
async def test_every_chunk_gets_a_vector_of_the_right_width(captured):
    """The ticket's actual deliverable: vectors in the pgvector column."""
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    _, chunks = await load(document_id)
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(len(chunk.embedding) == EMBEDDING_DIM for chunk in chunks)


@pytest.mark.anyio
async def test_embedding_model_is_recorded(captured):
    """Without this column a model change means re-embedding everything."""
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    _, chunks = await load(document_id)
    assert {chunk.embedding_model for chunk in chunks} == {"voyage-4"}


@pytest.mark.anyio
async def test_one_batched_call_per_document(captured):
    """
    "Batch-processes chunks" — not one API call per chunk.

    A real provider would still bill the same tokens, but 24 round trips
    instead of 1 is the difference between ingestion taking a second and
    taking half a minute.
    """
    fake = FakeEmbedder()
    document_id = await captured()

    await process_document(document_id, embedder=fake)

    _, chunks = await load(document_id)
    assert fake.calls == [("document", len(chunks))]


@pytest.mark.anyio
async def test_normalization_results_are_persisted(captured):
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    document, _ = await load(document_id)
    assert document.language in {"uk", "en"}
    assert document.content_hash and len(document.content_hash) == 64


@pytest.mark.anyio
async def test_offsets_index_into_the_stored_content(captured):
    """
    content[start_char:end_char] == chunk.content, exactly.

    This is what citations rely on. Note the offsets refer to the NORMALIZED
    text now stored in Document.content, not to the raw capture.
    """
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    document, chunks = await load(document_id)
    for chunk in chunks:
        assert document.content[chunk.start_char : chunk.end_char] == chunk.content


@pytest.mark.anyio
async def test_chunk_indexes_are_dense_and_ordered(captured):
    document_id = await captured()

    await process_document(document_id, embedder=FakeEmbedder())

    _, chunks = await load(document_id)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


# failure


@pytest.mark.anyio
async def test_embedding_failure_marks_the_document_failed(captured):
    document_id = await captured()

    await process_document(document_id, embedder=BrokenEmbedder())

    document, _ = await load(document_id)
    assert document.status is IngestionStatus.FAILED
    assert "Voyage unavailable" in document.error_message


@pytest.mark.anyio
async def test_embedding_failure_leaves_no_chunks_behind(captured):
    """
    The rollback in _record_failure.

    Without it the chunks written before the embedder failed would be
    committed with NULL vectors: rows that vector search returns but cannot
    rank, attached to a document reading FAILED.
    """
    document_id = await captured()

    await process_document(document_id, embedder=BrokenEmbedder())

    async with AsyncSessionLocal() as db:
        count = (
            await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document_id))
        ).scalar_one()
    assert count == 0


@pytest.mark.anyio
async def test_text_too_short_is_rejected_with_a_reason(captured):
    document_id = await captured(text="Too short.")

    await process_document(document_id, embedder=FakeEmbedder())

    document, chunks = await load(document_id)
    assert document.status is IngestionStatus.FAILED
    assert "too short" in document.error_message.lower()
    assert chunks == []


@pytest.mark.anyio
async def test_process_document_never_raises(captured):
    """
    Nothing awaits this coroutine in production, so an exception escaping it
    is swallowed and the document sits at PROCESSING forever.
    """
    document_id = await captured()

    await process_document(document_id, embedder=BrokenEmbedder())  # must not raise


@pytest.mark.anyio
async def test_missing_document_is_not_an_error():
    await process_document(uuid.uuid4(), embedder=FakeEmbedder())  # must not raise


# re-running


@pytest.mark.anyio
async def test_a_failed_document_can_be_retried(captured):
    """
    Regression: ck_documents_error_message_iff_failed rejects a row with
    status=processing while error_message still holds the last failure, so
    the retry used to die inside the un-protected first commit.
    """
    document_id = await captured()
    await process_document(document_id, embedder=BrokenEmbedder())

    await process_document(document_id, embedder=FakeEmbedder())

    document, chunks = await load(document_id)
    assert document.status is IngestionStatus.COMPLETED
    assert document.error_message is None
    assert all(chunk.embedding is not None for chunk in chunks)


@pytest.mark.anyio
async def test_reprocessing_replaces_chunks_rather_than_duplicating(captured):
    """
    uq_chunks_document_index would reject a second chunk 0, so re-running
    must delete before it inserts. This is the path a re-chunk after a
    DEFAULT_CHUNK_SIZE change will take.
    """
    document_id = await captured()
    await process_document(document_id, embedder=FakeEmbedder())
    _, first = await load(document_id)

    await process_document(document_id, embedder=FakeEmbedder())
    _, second = await load(document_id)

    assert len(second) == len(first)
    assert {chunk.id for chunk in second}.isdisjoint({chunk.id for chunk in first})


@pytest.mark.anyio
async def test_normalization_is_idempotent_across_runs(captured):
    """
    The pipeline overwrites content with its normalized form, so a re-run
    normalizes already-normalized text. If that were not stable, the hash
    would change on every pass and duplicate detection would be useless.
    """
    document_id = await captured()
    await process_document(document_id, embedder=FakeEmbedder())
    first, _ = await load(document_id)
    first_hash = first.content_hash

    await process_document(document_id, embedder=FakeEmbedder())
    second, _ = await load(document_id)

    assert second.content_hash == first_hash