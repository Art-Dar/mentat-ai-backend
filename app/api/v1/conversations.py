import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.models import Conversation, Message
from app.schemas.answer import RetrievalRead
from app.schemas.ask import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationRead,
    ConversationRename,
    MessageRead,
)
from app.services.conversations.store import (
    MAX_LIST_LIMIT,
    StoredRetrieval,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)

router = APIRouter()


def _summary(conversation: Conversation, message_count: int) -> ConversationRead:
    return ConversationRead(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=message_count,
    )


def _message(message: Message) -> MessageRead:
    """
    The snapshot is validated back through RetrievalRead rather than
    passed through as raw JSON: it was written by an older version of
    this code and will one day have been written by a much older one, so
    a shape that no longer matches should fail here where it can be seen
    rather than in the extension.
    """
    stored = StoredRetrieval.from_json(message.retrieval)
    return MessageRead(
        id=message.id,
        role=message.role.value,
        content=message.content,
        search_query=message.search_query,
        retrieval=(RetrievalRead.model_validate(stored.retrieval) if stored is not None else None),
        cited_markers=stored.cited_markers if stored is not None else [],
        created_at=message.created_at,
    )


async def _own_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation:
    """
    The caller's own thread, or a 404.
    """
    conversation = await get_conversation(db, conversation_id, user_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such conversation",
        )
    return conversation


@router.post(
    "/conversations",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
)
async def start_conversation(
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> ConversationRead:
    """
    Start an empty thread.

    The title may be omitted and usually is: it is filled from the first
    question when that arrives, which is a better label than anything a
    user types up front and costs no model call.
    """
    conversation = await create_conversation(db, user_id=user_id, title=payload.title)
    return _summary(conversation, message_count=0)


@router.get("/conversations", response_model=ConversationList)
async def get_conversations(
    limit: int = Query(default=30, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> ConversationList:
    """The caller's threads, most recent activity first."""
    conversations, total = await list_conversations(db, user_id=user_id, limit=limit, offset=offset)
    return ConversationList(
        # message_count is left at zero here rather than counted per row:
        # the sidebar shows titles and times, and counting would cost a
        # query per conversation for a number nobody reads. The detail
        # view has the real count.
        conversations=[_summary(c, message_count=0) for c in conversations],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def read_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> ConversationDetail:
    """
    One thread with every turn, including the citations each answer was
    built from at the time it was written.
    """
    conversation = await _own_conversation(db, conversation_id, user_id)
    messages = sorted(conversation.messages, key=lambda m: m.created_at)

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        messages=[_message(message) for message in messages],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationRead)
async def patch_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationRename,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> ConversationRead:
    conversation = await _own_conversation(db, conversation_id, user_id)
    message_count = len(conversation.messages)
    conversation = await rename_conversation(db, conversation, payload.title)
    return _summary(conversation, message_count=message_count)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> None:
    """
    Delete a thread and its messages.
    """
    conversation = await _own_conversation(db, conversation_id, user_id)
    await delete_conversation(db, conversation)
