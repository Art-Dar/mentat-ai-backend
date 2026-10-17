import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Conversation, Message, MessageRole
from app.schemas.answer import RetrievalRead
from app.services.answering.answerer import Answer
from app.services.answering.answering_prompt import Turn

# The column holds 200; a title longer than this is unreadable in a
# sidebar anyway, and the first clause of a question is almost always
# the recognisable part.
TITLE_CHARS = 80

# A page of conversations. The sidebar shows a handful; this is the
# ceiling so a client cannot ask for the whole history in one request.
MAX_LIST_LIMIT = 100


@dataclass(frozen=True)
class StoredRetrieval:
    """
    What an assistant message carries beside its prose.
    """

    retrieval: dict
    cited_markers: list[str]
    model: str

    def to_json(self) -> dict:
        return {
            "retrieval": self.retrieval,
            "cited_markers": self.cited_markers,
            "model": self.model,
        }

    @classmethod
    def from_json(cls, payload: dict | None) -> "StoredRetrieval | None":
        if not payload:
            return None
        return cls(
            retrieval=payload.get("retrieval") or {},
            cited_markers=list(payload.get("cited_markers") or []),
            model=payload.get("model") or "",
        )


def derive_title(question: str) -> str:
    """The first question, trimmed. No model call for a sidebar label."""
    flat = " ".join(question.split())
    if len(flat) <= TITLE_CHARS:
        return flat
    return flat[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"


async def create_conversation(
    db: AsyncSession, *, user_id: uuid.UUID, title: str | None = None
) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation | None:
    """
    Ownership is part of the WHERE rather than checked afterwards, so
    there is no path where a row is loaded and the check is forgotten.
    selectinload because the messages are read outside the lazy-load
    boundary — touching them later raises MissingGreenlet under asyncio.
    """
    return (
        (
            await db.execute(
                select(Conversation)
                .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .options(selectinload(Conversation.messages))
            )
        )
        .scalars()
        .one_or_none()
    )


async def list_conversations(
    db: AsyncSession, *, user_id: uuid.UUID, limit: int = 30, offset: int = 0
) -> tuple[list[Conversation], int]:
    """
    A page of the caller's conversations, newest activity first, and the
    total so a client can page without guessing when to stop.
    """
    limit = max(1, min(limit, MAX_LIST_LIMIT))

    rows = (
        (
            await db.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )

    total = (
        await db.execute(
            select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        )
    ).scalar_one()

    return list(rows), total


async def rename_conversation(
    db: AsyncSession, conversation: Conversation, title: str
) -> Conversation:
    conversation.title = title.strip()[:200]
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def delete_conversation(db: AsyncSession, conversation: Conversation) -> None:
    """
    Deletes the thread and its messages; touches no documents.
    """
    await db.execute(delete(Conversation).where(Conversation.id == conversation.id))
    await db.commit()


def history_for(conversation: Conversation) -> tuple[Turn, ...]:
    """
    Returns every turn; the prompt and the condenser each take the tail
    they want. Trimming here would mean two different ideas of "recent"
    depending on which caller asked.
    """
    return tuple(
        Turn(role=message.role.value, content=message.content)  # type: ignore[arg-type]
        for message in sorted(conversation.messages, key=lambda m: m.created_at)
    )


async def append_turn(db: AsyncSession, conversation: Conversation, answer: Answer) -> Message:
    """
    Write the question and its answer, and return the assistant message.
    """
    question = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=answer.question,
        # Only when the question was rewritten. Storing it unconditionally
        # would duplicate `content` on every first turn for no reason.
        search_query=(answer.search_query if answer.search_query != answer.question else None),
    )
    reply = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=answer.text,
        retrieval=StoredRetrieval(
            retrieval=RetrievalRead.from_context(answer.context).model_dump(mode="json"),
            cited_markers=[citation.marker for citation in answer.cited],
            model=answer.model,
        ).to_json(),
    )
    db.add_all([question, reply])

    # Adding a message does not touch the conversation ROW, so without a
    # statement against it the sidebar ordering freezes at creation time
    # and the thread you have been using all afternoon sinks to the
    # bottom. Issuing the UPDATE is the part that matters.
    #
    # `updated_at` is then set explicitly even though Base declares
    # onupdate=func.now() and would fill it anyway — measured, an
    # update() carrying no values at all still emits
    # "SET updated_at=now()". Being explicit costs nothing and means the
    # ordering of this feature does not silently depend on a default
    # declared on a base class shared with every other table.
    values: dict = {"updated_at": func.now()}
    if not conversation.title:
        values["title"] = derive_title(answer.question)

    await db.execute(
        update(Conversation).where(Conversation.id == conversation.id).values(**values)
    )
    await db.commit()
    await db.refresh(reply)
    return reply


__all__ = [
    "MAX_LIST_LIMIT",
    "TITLE_CHARS",
    "StoredRetrieval",
    "append_turn",
    "create_conversation",
    "delete_conversation",
    "derive_title",
    "get_conversation",
    "history_for",
    "list_conversations",
    "rename_conversation",
]
