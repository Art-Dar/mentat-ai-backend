import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.base import RequestSchema
from app.services.retrieval.search import DEFAULT_TOP_K, MAX_TOP_K

MAX_QUERY_CHARS = 1000

STRICT = ConfigDict(extra="forbid")


class SearchRequest(RequestSchema):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Narrow to specific documents — "search within this article", or a
    # conversation already pinned to a few sources. Always intersected
    # with the caller's own documents, never a way to reach past them.
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=MAX_TOP_K)


class SearchHitRead(BaseModel):
    model_config = STRICT

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    # Cosine similarity, 1.0 being identical
    # Returned rather than a rank so the client can show "weak match"
    similarity: float
    document_title: str | None
    document_url: str | None


class SearchResponse(BaseModel):
    model_config = STRICT

    query: str
    top_k: int
    count: int
    hits: list[SearchHitRead]