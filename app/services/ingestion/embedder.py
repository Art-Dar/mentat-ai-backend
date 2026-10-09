import asyncio
import logging
from typing import Protocol

import voyageai
from voyageai.error import (
    APIConnectionError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
)

from app.core.config import settings
from app.models.chunk import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# the SDK's default is 128 texts per request - failed request cheap to retry.
BATCH_SIZE = 128

_RETRYABLE = (
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    APIConnectionError,
)

MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0


class EmbeddingError(RuntimeError):
    """Embedding could not be produced. Recorded on the document row."""

class Embedder(Protocol):
    """What the pipeline needs. Implemented by Voyage and by the test fake."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    """Embedder backed by the Voyage API."""

    def __init__(
            self,
            client: voyageai.AsyncClient | None = None,
            model: str | None = None,
            dimension: int = EMBEDDING_DIM,
            batch_size: int = BATCH_SIZE,
    ) -> None:
        # The client is injectable so tests can pass a stub without patching.
        if client is None:
            if not settings.voyage_api_key:
                # Caught here, not letting the SDK send an unauthenticated
                # request (missing setting instead of 401 from else's service)
                raise EmbeddingError("VOYAGE_API_KEY is not set")
            client = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self._client = client
        self._model = model or settings.embedding_model
        self._dimension = dimension
        self._batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start: start + self._batch_size]
            vectors.extend(await self._embed(batch, input_type="document"))

        if len(vectors) != len(texts):
            raise EmbeddingError(f"Voyage returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a question for retrieval. Note the different input_type."""
        vectors = await self._embed([text], input_type="query")
        return vectors[0]

    async def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = await self._client.embed(
                    texts,
                    model=self._model,
                    input_type=input_type,
                    output_dimension=self._dimension,
                    # truncation=True (the default) trims an over-long text
                    # rather than failing the whole batch for one chunk.
                    truncation=True,
                )
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "voyage %s (attempt %d/%d), retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # auth, malformed request, anything else
                raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc
            else:
                return self._validated(result.embeddings, expected=len(texts))

        raise EmbeddingError(
            f"Voyage unavailable after {MAX_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _validated(self, vectors: list[list[float]], *, expected: int) -> list[list[float]]:
        """
        Check the shape before it reaches Postgres.
        """
        if len(vectors) != expected:
            raise EmbeddingError(f"expected {expected} vectors, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    f"expected {self._dimension}-dim vectors, got {len(vector)} — "
                    f"check output_dimension is being passed"
                )
        return vectors


def get_embedder() -> Embedder:
    """The embedder the pipeline uses. Replace in tests via dependency passing."""
    return VoyageEmbedder()
