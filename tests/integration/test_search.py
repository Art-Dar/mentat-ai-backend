"""
Vector similarity search against a real pgvector index.
Integration tests. FakeEmbedder's vectors are assertable without a real model:
"""

import uuid

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.dialects import postgresql

from app.core.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentSource, User
from app.models.chunk import EMBEDDING_DIM
from app.models.user import AuthIdentity, AuthProvider
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.ingestion.pipeline import process_document
from app.services.retrieval.search import (
    MAX_TOP_K,
    SearchHit,
    _build_query,
    search_by_text,
    search_chunks,
)

# Each article must produce SEVERAL chunks
# MIN_CHUNKS guards it: with one chunk per document,
# "top_k caps the results" and "ordered by similarity"
# are both trivially true and the tests can never fail.
MIN_CHUNKS_PER_DOCUMENT = 2

ARTICLE_A = "\n\n".join(
    [
        "Vector databases store high-dimensional embeddings and retrieve them by "
        "approximate nearest neighbour search. The HNSW index builds a navigable "
        "small-world graph over the vectors, so recall stays high without scanning "
        "every row in the table at query time. Unlike IVFFlat it needs no training "
        "pass over existing data, so it is correct even on an empty table and stays "
        "correct as rows are added one capture at a time by the user.",
        "Chunking strategy matters more than most teams expect. If a chunk is too "
        "large, one relevant sentence is diluted by a thousand characters of "
        "unrelated prose and the similarity score for a precise question drops. If "
        "it is too small, it loses the context needed to make sense on its own when "
        "it comes back as a citation to the reader, who has no way to tell what the "
        "surrounding paragraph said.",
        "Нормалізація тексту виконується перед розбиттям на фрагменти. Вона "
        "приводить пробіли до єдиного вигляду, розкодовує HTML-сутності та "
        "застосовує форму NFKC до символів Юнікоду. Без цього однакові за змістом "
        "тексти дають різні вектори, і дублікати неможливо виявити за хешем вмісту. "
        "Визначення мови виконується лише для подальших етапів обробки.",
        "Asymmetric embedding means documents and queries are encoded with "
        "different instructions. A stored passage is long and declarative while a "
        "question is short and interrogative. Encoding both the same way places "
        "them in subtly different regions of the vector space, and retrieval "
        "quality degrades quietly rather than failing loudly in an obvious way.",
    ]
)

ARTICLE_B = "\n\n".join(
    [
        "Espresso extraction depends on grind size, dose and water temperature. A "
        "shot pulled too fast tastes sour because the water has not had time to "
        "dissolve the sugars, while one pulled too slow turns bitter as the water "
        "strips compounds that should have stayed locked inside the puck. Adjusting "
        "the grinder is almost always the first thing to try.",
        "Milk texturing is a separate skill from extraction entirely. The goal is "
        "microfoam, where the bubbles are too small to see individually, and that "
        "comes from introducing air only in the first second or two and then "
        "spinning the jug to break down whatever larger bubbles did manage to form "
        "before the surface of the milk closed over the steam wand tip.",
        "Water chemistry is the variable most people never touch. Too soft and the "
        "shot tastes flat and hollow; too hard and the machine scales up within "
        "months. Bottled water sold specifically for espresso exists for exactly "
        "this reason, and in most cities a cheap filter jug is enough to bring tap "
        "water into a workable range for daily brewing at home.",
    ]
)


@pytest.fixture
async def search_user(keep_rows):
    email = f"search-{uuid.uuid4().hex[:8]}@example.com"
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
async def corpus(search_user, keep_rows):
    """
    Two fully ingested documents on unrelated topics.

    Two rather than one so the document_ids filter has something to
    exclude, and so a search can return hits from more than one source.
    """
    ids: list[uuid.UUID] = []

    for article in (ARTICLE_A, ARTICLE_B):
        async with AsyncSessionLocal() as db:
            document = Document(
                user_id=search_user,
                source=DocumentSource.ARTICLE,
                content=article,
                url=f"https://example.com/{uuid.uuid4().hex[:6]}",
                title="Test article",
            )
            db.add(document)
            await db.commit()
            ids.append(document.id)
        await process_document(ids[-1], embedder=FakeEmbedder())

    yield ids

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(ids)))
        await db.commit()


async def all_chunks(document_ids: list[uuid.UUID]) -> list[Chunk]:
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Chunk)
                .where(Chunk.document_id.in_(document_ids))
                .order_by(Chunk.document_id, Chunk.chunk_index)
            )
        ).scalars()
        return list(rows)


# the core contract


@pytest.mark.anyio
async def test_a_chunks_own_vector_retrieves_that_chunk_first(corpus):
    """
    The one relevance claim that holds with meaningless vectors: searching
    with a chunk's exact stored embedding must return that chunk at the
    top, with similarity 1.0.
    """
    chunks = await all_chunks(corpus)
    target = chunks[len(chunks) // 2]

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(db, list(target.embedding), top_k=3)

    assert hits[0].chunk_id == target.id
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


@pytest.mark.anyio
async def test_results_are_ordered_by_descending_similarity(corpus):
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(
            db, list(chunks[0].embedding), top_k=MAX_TOP_K, min_similarity=-1.0
        )

    scores = [hit.similarity for hit in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.anyio
async def test_similarity_is_not_distance(corpus):
    """
    `<=>` returns 0 for identical vectors. Returning it raw would mean the
    best hit scored lowest, and every caller's threshold would be inverted.
    """
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(db, list(chunks[0].embedding), top_k=5, min_similarity=-1.0)

    assert hits[0].similarity > hits[-1].similarity
    assert all(-1.0001 <= hit.similarity <= 1.0001 for hit in hits)


@pytest.mark.anyio
async def test_top_k_caps_the_result_count(corpus):
    chunks = await all_chunks(corpus)
    assert len(chunks) >= 2 * MIN_CHUNKS_PER_DOCUMENT  # or this proves nothing

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(db, list(chunks[0].embedding), top_k=2, min_similarity=-1.0)

    assert len(hits) == 2


@pytest.mark.anyio
async def test_hits_carry_everything_a_citation_needs(corpus):
    """No second query per result just to render the source."""
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hit = (await search_chunks(db, list(chunks[0].embedding), top_k=1))[0]

    assert isinstance(hit, SearchHit)
    assert hit.document_title == "Test article"
    assert hit.document_url.startswith("https://example.com/")
    assert hit.content
    assert hit.end_char > hit.start_char


@pytest.mark.anyio
async def test_offsets_returned_by_search_still_index_the_document(corpus):
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(
            db, list(chunks[0].embedding), top_k=MAX_TOP_K, min_similarity=-1.0
        )
        for hit in hits:
            document = await db.get(Document, hit.document_id)
            assert document.content[hit.start_char : hit.end_char] == hit.content


# filters


@pytest.mark.anyio
async def test_document_ids_scopes_the_search(corpus):
    first, second = corpus
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(
            db,
            list(chunks[0].embedding),
            top_k=MAX_TOP_K,
            document_ids=[second],
            min_similarity=-1.0,
        )

    assert hits
    assert {hit.document_id for hit in hits} == {second}
    assert first not in {hit.document_id for hit in hits}


@pytest.mark.anyio
async def test_empty_document_ids_searches_nothing(corpus):
    """Distinct from None, which searches everything."""
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(db, list(chunks[0].embedding), document_ids=[])

    assert hits == []


@pytest.mark.anyio
async def test_min_similarity_drops_weak_matches(corpus):
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        # Nothing can beat 1.0, so only the exact self-match survives.
        hits = await search_chunks(
            db, list(chunks[0].embedding), top_k=MAX_TOP_K, min_similarity=0.999
        )

    assert len(hits) == 1
    assert hits[0].chunk_id == chunks[0].id


@pytest.mark.anyio
async def test_the_default_floor_drops_negative_similarity(corpus):
    """
    A negative cosine means actively dissimilar, not merely weak, so the
    default min_similarity of 0.0 removes it. Passing -1.0 turns the floor
    off, which is what the ordering and top_k tests above do so that they
    test one thing each.
    """
    chunks = await all_chunks(corpus)

    async with AsyncSessionLocal() as db:
        unfiltered = await search_chunks(
            db, list(chunks[0].embedding), top_k=MAX_TOP_K, min_similarity=-1.0
        )
        defaulted = await search_chunks(db, list(chunks[0].embedding), top_k=MAX_TOP_K)

    assert any(hit.similarity < 0 for hit in unfiltered)  # else nothing to drop
    assert len(defaulted) < len(unfiltered)
    assert all(hit.similarity >= 0 for hit in defaulted)


@pytest.mark.anyio
async def test_chunks_without_embeddings_are_never_returned(corpus):
    """
    They exist between chunking and embedding. A NULL distance sorts last
    rather than erroring, so without the filter they would appear as silent
    junk at the bottom of a short result set.
    """
    chunks = await all_chunks(corpus)
    async with AsyncSessionLocal() as db:
        orphan = Chunk(
            document_id=corpus[0],
            chunk_index=9_999,
            content="a chunk that was never embedded",
            start_char=0,
            end_char=31,
        )
        db.add(orphan)
        await db.commit()
        orphan_id = orphan.id

    async with AsyncSessionLocal() as db:
        hits = await search_chunks(
            db, list(chunks[0].embedding), top_k=MAX_TOP_K, min_similarity=-1.0
        )

    assert orphan_id not in {hit.chunk_id for hit in hits}


# validation


@pytest.mark.anyio
async def test_wrong_dimension_is_rejected_before_the_query():
    async with AsyncSessionLocal() as db:
        with pytest.raises(ValueError, match="dimensions"):
            await search_chunks(db, [0.1] * 128)


@pytest.mark.anyio
async def test_top_k_is_bounded():
    async with AsyncSessionLocal() as db:
        with pytest.raises(ValueError, match="top_k"):
            await search_chunks(db, [0.1] * EMBEDDING_DIM, top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            await search_chunks(db, [0.1] * EMBEDDING_DIM, top_k=MAX_TOP_K + 1)


# the text entry point


@pytest.mark.anyio
async def test_search_by_text_uses_the_query_input_type(corpus):
    """
    Documents and queries are embedded asymmetrically. Calling
    embed_documents here would not raise — it would just make retrieval
    quietly worse — so the call is pinned by a test instead.
    """
    fake = FakeEmbedder()

    async with AsyncSessionLocal() as db:
        await search_by_text(db, "how does HNSW work", embedder=fake)

    assert fake.calls == [("query", 1)]


@pytest.mark.anyio
async def test_blank_query_makes_no_api_call(corpus):
    fake = FakeEmbedder()

    async with AsyncSessionLocal() as db:
        assert await search_by_text(db, "   ", embedder=fake) == []

    assert fake.calls == []


# the index


@pytest.mark.anyio
async def test_the_query_emits_the_cosine_operator():
    """
    The module must emit `<=>`, the operator ix_chunks_embedding_hnsw
    indexes with vector_cosine_ops.
    """
    compiled = str(
        _build_query([0.01] * EMBEDDING_DIM, 5, None).compile(dialect=postgresql.dialect())
    )

    assert "<=>" in compiled
    assert "<->" not in compiled  # l2_distance
    assert "<#>" not in compiled  # max_inner_product


@pytest.mark.anyio
async def test_the_query_can_use_the_hnsw_index(corpus):
    """
    On a handful of rows the planner picks a sequential scan because that
    is genuinely cheaper, so the default plan proves nothing. Forcing
    seqscan off is what shows the index is REACHABLE — if it is not, the
    query's distance operator does not match vector_cosine_ops and every
    search on a real corpus will scan the whole table.
    """
    vector = "[" + ",".join(["0.01"] * EMBEDDING_DIM) + "]"

    async with AsyncSessionLocal() as db:
        await db.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            await db.execute(
                text(
                    "EXPLAIN SELECT id FROM chunks "
                    "WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:qv AS vector) LIMIT 5"
                ),
                {"qv": vector},
            )
        ).scalars()

    assert any("ix_chunks_embedding_hnsw" in line for line in plan)