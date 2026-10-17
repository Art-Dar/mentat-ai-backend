import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.user import User


def _enum_values(enum_class) -> list[str]:
    return [member.value for member in enum_class]


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Conversation(Base):
    """
    A thread of questions against the knowledge base.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        # The list view is "my conversations, most recent first", so the
        # index matches that query exactly.
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(200))

    user: Mapped["User"] = relationship()
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        # A user turn has nothing retrieved; an assistant turn always
        # does, even when retrieval found nothing (an empty object, not
        # NULL). Enforced here because a missing snapshot is invisible
        # until someone reopens the conversation weeks later.
        CheckConstraint(
            "(role = 'assistant') OR (retrieval IS NULL)",
            name="ck_messages_retrieval_only_on_assistant",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, name="message_role", values_callable=_enum_values)
    )
    content: Mapped[str] = mapped_column(Text)

    # What was actually embedded for this turn. Only set on the user
    # message of a follow-up, where it differs from `content` because the
    # question was condensed against the history.
    search_query: Mapped[str | None] = mapped_column(Text)

    retrieval: Mapped[dict | None] = mapped_column(JSONB)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
