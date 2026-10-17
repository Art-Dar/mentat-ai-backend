import logging
import math

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.chunk import EMBEDDING_DIM

logger = logging.getLogger(__name__)

MODEL_NAME = "lexical-tfidf-svd"


class LexicalEmbedder:
    """
    `fit` must be called before anything else, with the whole corpus.
    That is the one awkward difference from a hosted embedder, and it is
    inherent: TF-IDF weights are a property of the corpus, so there is
    no such thing as embedding a document without one. A sweep fits once
    per chunking and reuses it, which is also what makes the sweep fast.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.calls: list[tuple[str, int]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, corpus: list[str]) -> "LexicalEmbedder":
        if not corpus:
            raise ValueError("cannot fit a lexical embedder on an empty corpus")

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
        )
        matrix = self._vectorizer.fit_transform(corpus)

        # SVD needs strictly fewer components than features and samples.
        # A tiny corpus is the normal case in a student project, so this
        # clamps rather than raising — the vectors are zero-padded up to
        # EMBEDDING_DIM afterwards so the column width always matches.
        components = min(self.dimension, matrix.shape[1] - 1, matrix.shape[0] - 1)
        if components < 1:
            raise ValueError(
                f"corpus of {matrix.shape[0]} document(s) is too small to fit; "
                "at least 2 are needed"
            )
        if components < self.dimension:
            logger.info(
                "lexical embedder: %d components from a %d-document corpus "
                "(padded to %d)",
                components,
                matrix.shape[0],
                self.dimension,
            )

        self._svd = TruncatedSVD(n_components=components, random_state=0)
        self._svd.fit(matrix)
        return self

    def _project(self, texts: list[str]) -> list[list[float]]:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("fit() must be called before embedding")

        reduced = self._svd.transform(self._vectorizer.transform(texts))

        vectors: list[list[float]] = []
        for row in reduced:
            padded = np.zeros(self.dimension, dtype=np.float64)
            padded[: len(row)] = row
            # Unit length, so the cosine distance pgvector computes
            # behaves. A zero vector (text sharing no n-gram with the
            # corpus) would divide by zero, so it stays zero — maximally
            # distant from everything, which is the honest answer.
            norm = math.sqrt(float(np.dot(padded, padded)))
            vectors.append((padded / norm).tolist() if norm else padded.tolist())
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("document", len(texts)))
        return self._project(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", 1))
        return self._project([text])[0]


__all__ = ["MODEL_NAME", "LexicalEmbedder"]
