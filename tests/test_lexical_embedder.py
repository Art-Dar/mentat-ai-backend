import math

import pytest

from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.lexical_embedder import LexicalEmbedder

CORPUS = [
    "Payment terms: net 14 days from the invoice date, two percent early.",
    "Green coffee should be stored between 15 and 20 degrees Celsius.",
    "Roasted coffee is not returnable once the vacuum seal has been broken.",
    "Sprint planning meeting notes: streaming deferred to sprint three.",
    "The warehouse move improved lead times to four working days.",
    "Delivery within the city is included for orders above 2000 UAH.",
]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture
def embedder():
    return LexicalEmbedder().fit(CORPUS)


@pytest.mark.anyio
async def test_vectors_are_the_width_the_column_expects(embedder):
    """
    A dimension mismatch is a database error at insert time, which is
    the right place for it to fail — but only if the embedder never
    quietly returns a short vector.
    """
    vectors = await embedder.embed_documents(CORPUS)

    assert len(vectors) == len(CORPUS)
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


@pytest.mark.anyio
async def test_vectors_are_unit_length(embedder):
    """
    pgvector's cosine operator needs this to behave. A non-normalised
    vector still returns a ranking, just a subtly wrong one.
    """
    for vector in await embedder.embed_documents(CORPUS):
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.anyio
async def test_the_same_text_always_gives_the_same_vector(embedder):
    """A sweep that re-embeds between settings must not also be
    measuring embedder noise."""
    first = await embedder.embed_query("when is the invoice due?")
    second = await embedder.embed_query("when is the invoice due?")

    assert first == second


def test_embedding_before_fitting_is_an_error():
    """
    The one awkward difference from a hosted embedder, and it is
    inherent: TF-IDF weights are a property of the corpus, so there is
    no such thing as embedding without one. Better loud than silently
    returning zeros, which would rank everything equally.
    """
    with pytest.raises(RuntimeError, match="fit"):
        LexicalEmbedder()._project(["anything"])


def test_fitting_on_nothing_is_refused():
    with pytest.raises(ValueError, match="empty"):
        LexicalEmbedder().fit([])


def test_a_corpus_of_one_is_refused():
    """
    SVD needs fewer components than samples. One document would
    otherwise produce a degenerate projection rather than an error.
    """
    with pytest.raises(ValueError, match="too small"):
        LexicalEmbedder().fit(["only one document"])


@pytest.mark.anyio
async def test_a_question_lands_nearest_the_passage_that_answers_it(embedder):
    """
    The test that separates this from FakeEmbedder. If this fails, a
    tuning sweep run against it is measuring noise and every number it
    produces is meaningless.
    """
    vectors = await embedder.embed_documents(CORPUS)
    query = await embedder.embed_query("how should green coffee be stored?")

    ranked = sorted(
        range(len(CORPUS)), key=lambda i: cosine(query, vectors[i]), reverse=True
    )
    assert "15 and 20 degrees" in CORPUS[ranked[0]]


@pytest.mark.anyio
async def test_related_text_scores_above_unrelated_text(embedder):
    vectors = await embedder.embed_documents(CORPUS)
    query = await embedder.embed_query("when do I have to pay the invoice?")

    payment = cosine(query, vectors[0])
    warehouse = cosine(query, vectors[4])

    assert payment > warehouse


@pytest.mark.anyio
async def test_text_sharing_nothing_with_the_corpus_is_not_a_crash(embedder):
    """
    A zero vector is the honest answer — maximally distant from
    everything — and it must not divide by zero on its way out.
    """
    vector = await embedder.embed_query("日本語のまったく無関係な文章")

    assert len(vector) == EMBEDDING_DIM
    assert all(math.isfinite(x) for x in vector)


@pytest.mark.anyio
async def test_a_small_corpus_is_padded_rather_than_refused(embedder):
    """
    A student project's corpus is routinely smaller than the embedding
    dimension. The vectors are padded so the column width still matches
    rather than the whole thing failing.
    """
    small = LexicalEmbedder().fit(CORPUS[:3])
    vectors = await small.embed_documents(["payment terms and invoices"])

    assert len(vectors[0]) == EMBEDDING_DIM


@pytest.mark.anyio
async def test_calls_are_recorded_like_the_other_embedders(embedder):
    """Matches FakeEmbedder's interface so tests can swap either in."""
    await embedder.embed_documents(["a", "b"])
    await embedder.embed_query("q")

    assert embedder.calls == [("document", 2), ("query", 1)]
