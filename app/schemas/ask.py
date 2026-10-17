import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.answer import RetrievalRead
from app.schemas.base import RequestSchema
from app.services.retrieval.search import DEFAULT_TOP_K, MAX_TOP_K

STRICT = ConfigDict(extra="forbid")

# Long enough for a real question, short enough that nobody pastes a
# document in and pays to embed it as a query.
MAX_QUESTION_CHARS = 1000

MAX_TITLE_CHARS = 200


class AskRequest(RequestSchema):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    # Omitted, the question is answered standalone and
    # nothing is written — which is what the browser extension's
    # quick-lookup wants, and it keeps /ask usable without first creating
    # a conversation.
    conversation_id: uuid.UUID | None = None

    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Ask within specific documents — "what does this invoice say?".
    # Always intersected with the caller's own, never a way past them.
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=MAX_TOP_K)


class AskResponse(BaseModel):
    model_config = STRICT

    answer: str
    question: str
    search_query: str

    # Which markers the prose actually references
    cited_markers: list[str]
    # False when the answer references no source at all. Not an error:
    # "I could not find that in your documents" is correct and cites
    # nothing. It means the client should not present this as verifiable.
    grounded: bool

    retrieval: RetrievalRead
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    dropped_sources: int = 0
    # The model stopped at its token limit, so the prose may end
    # mid-sentence.
    truncated: bool = False


class ConversationCreate(RequestSchema):
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)


class ConversationRename(RequestSchema):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)


class MessageRead(BaseModel):
    model_config = STRICT

    id: uuid.UUID
    role: str
    content: str
    # Only on a user message whose question was condensed.
    search_query: str | None = None
    # Only on an assistant message.
    retrieval: RetrievalRead | None = None
    cited_markers: list[str] = []
    created_at: datetime.datetime


class ConversationRead(BaseModel):
    model_config = STRICT

    id: uuid.UUID
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    message_count: int


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class ConversationList(BaseModel):
    model_config = STRICT

    conversations: list[ConversationRead]
    total: int
    limit: int
    offset: int
