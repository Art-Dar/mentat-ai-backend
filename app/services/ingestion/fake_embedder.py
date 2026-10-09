# deterministic Embedder for tests and local runs without an API key.

import hashlib
import math

from app.models.chunk import EMBEDDING_DIM


class FakeEmbedder:
    # deterministic (same text -> same vector, tests assert equality)
    # unit-length (cosine distance behaves)
    # correctly dimensioned (column-width mistake still fails as in production)
    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.calls: list[tuple[str, int]] = []  # (input_type, batch size)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("document", len(texts)))
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", 1))
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] + i) % 256 - 128 for i in range(self.dimension)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]
