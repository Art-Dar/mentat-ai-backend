# Nothing here touches the network:
# VoyageEmbedder takes an injectable client,so batching, retry and shape validation
# are all exercised against stubs.


import pytest
import voyageai.error as voyage_error

from app.services.ingestion.embedder import (
    BATCH_SIZE,
    MAX_ATTEMPTS,
    EmbeddingError,
    VoyageEmbedder,
)
from app.services.ingestion.fake_embedder import FakeEmbedder

DIM = 8


class StubResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.total_tokens = 0


class StubClient:
    """Records every call and returns correctly shaped vectors."""

    def __init__(self, dimension: int = DIM):
        self.dimension = dimension
        self.calls: list[dict] = []

    async def embed(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return StubResult([[0.1] * self.dimension for _ in texts])


class FailingClient:
    """Raises `error` for the first `failures` calls, then succeeds."""

    def __init__(self, error: Exception, failures: int, dimension: int = DIM):
        self.error = error
        self.failures = failures
        self.dimension = dimension
        self.attempts = 0

    async def embed(self, texts, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return StubResult([[0.1] * self.dimension for _ in texts])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry backoff shouldn't make the suite slow."""

    async def instant(_seconds):
        return None

    monkeypatch.setattr("app.services.ingestion.embedder.asyncio.sleep", instant)


def embedder(client, **kwargs) -> VoyageEmbedder:
    return VoyageEmbedder(client=client, model="voyage-4", dimension=DIM, **kwargs)


# the parameters that must never be omitted


@pytest.mark.anyio
async def test_output_dimension_is_always_passed():
    client = StubClient()
    await embedder(client).embed_documents(["hello"])

    assert client.calls[0]["output_dimension"] == DIM


@pytest.mark.anyio
async def test_documents_and_queries_use_different_input_types():
    client = StubClient()
    emb = embedder(client)

    await emb.embed_documents(["stored text"])
    await emb.embed_query("a question")

    assert client.calls[0]["input_type"] == "document"
    assert client.calls[1]["input_type"] == "query"


@pytest.mark.anyio
async def test_model_is_passed_through():
    client = StubClient()
    await embedder(client).embed_documents(["hello"])
    assert client.calls[0]["model"] == "voyage-4"


# batching


@pytest.mark.anyio
async def test_long_input_is_split_into_batches():
    client = StubClient()
    texts = [f"chunk {i}" for i in range(5)]

    vectors = await embedder(client, batch_size=2).embed_documents(texts)

    assert len(client.calls) == 3  # 2 + 2 + 1
    assert [len(call["texts"]) for call in client.calls] == [2, 2, 1]
    assert len(vectors) == 5


@pytest.mark.anyio
async def test_batched_vectors_stay_in_input_order():
    class OrderedClient(StubClient):
        async def embed(self, texts, **kwargs):
            self.calls.append({"texts": list(texts), **kwargs})
            # encode each text's first character so order is checkable
            return StubResult([[float(ord(t[0]))] * self.dimension for t in texts])

    client = OrderedClient()
    texts = ["a", "b", "c", "d", "e"]

    vectors = await embedder(client, batch_size=2).embed_documents(texts)

    assert [v[0] for v in vectors] == [float(ord(t)) for t in texts]


@pytest.mark.anyio
async def test_empty_input_makes_no_api_call():
    client = StubClient()
    assert await embedder(client).embed_documents([]) == []
    assert client.calls == []


def test_default_batch_size_is_within_the_api_limit():
    # Voyage accepts at most 1000 texts per request.
    assert 0 < BATCH_SIZE <= 1000


# retry


@pytest.mark.anyio
async def test_transient_error_is_retried_then_succeeds():
    client = FailingClient(voyage_error.RateLimitError("slow down"), failures=2)

    vectors = await embedder(client).embed_documents(["hello"])

    assert client.attempts == 3
    assert len(vectors) == 1


@pytest.mark.anyio
async def test_retries_give_up_after_max_attempts():
    client = FailingClient(voyage_error.ServerError("boom"), failures=99)

    with pytest.raises(EmbeddingError, match="unavailable"):
        await embedder(client).embed_documents(["hello"])

    assert client.attempts == MAX_ATTEMPTS


@pytest.mark.anyio
async def test_auth_error_is_not_retried():
    """A bad API key will not fix itself; retrying only delays the failure."""
    client = FailingClient(voyage_error.AuthenticationError("bad key"), failures=99)

    with pytest.raises(EmbeddingError, match="AuthenticationError"):
        await embedder(client).embed_documents(["hello"])

    assert client.attempts == 1


# shape validation


@pytest.mark.anyio
async def test_wrong_dimension_is_caught_before_the_database():
    client = StubClient(dimension=1024)  # as if output_dimension were dropped

    with pytest.raises(EmbeddingError, match="output_dimension"):
        await embedder(client).embed_documents(["hello"])


@pytest.mark.anyio
async def test_missing_vectors_are_caught():
    class ShortClient(StubClient):
        async def embed(self, texts, **kwargs):
            return StubResult([[0.1] * self.dimension])  # one vector for N texts

    with pytest.raises(EmbeddingError):
        await embedder(ShortClient()).embed_documents(["a", "b", "c"])


# the fake used by pipeline tests (services/ingestion/fake_embedder)


@pytest.mark.anyio
async def test_fake_embedder_is_deterministic():
    fake = FakeEmbedder(dimension=DIM)

    first = await fake.embed_documents(["same text"])
    second = await fake.embed_documents(["same text"])

    assert first == second


@pytest.mark.anyio
async def test_fake_embedder_returns_correct_dimension():
    fake = FakeEmbedder(dimension=DIM)
    vectors = await fake.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)


@pytest.mark.anyio
async def test_fake_embedder_distinguishes_different_texts():
    fake = FakeEmbedder(dimension=DIM)
    vectors = await fake.embed_documents(["first", "second"])

    assert vectors[0] != vectors[1]


# configuration


@pytest.mark.anyio
async def test_missing_api_key_names_the_setting(monkeypatch):
    monkeypatch.setattr("app.services.ingestion.embedder.settings.voyage_api_key", "")

    with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()
