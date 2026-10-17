import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user_id
from app.schemas.answer import RetrievalRead
from app.schemas.ask import AskRequest, AskResponse
from app.services.answering.answerer import answer_question
from app.services.conversations.store import append_turn, get_conversation, history_for
from app.services.ingestion.embedder import Embedder, get_embedder
from app.services.llm.base import ChatClient, LlmError, LlmUnavailable
from app.services.llm.groq_client import get_chat_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    embedder: Embedder = Depends(get_embedder),
    chat: ChatClient = Depends(get_chat_client),
) -> AskResponse:
    conversation = None
    history: tuple = ()

    if payload.conversation_id is not None:
        conversation = await get_conversation(db, payload.conversation_id, user_id)
        # The same 404 for "not yours" as for "does not exist". A 403
        # would confirm the id is real, and a conversation id is an
        # enumeration oracle over what other people have been asking.
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No such conversation",
            )
        history = history_for(conversation)

    try:
        answer = await answer_question(
            db,
            question=payload.question,
            user_id=user_id,
            embedder=embedder,
            chat=chat,
            history=history,
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
            document_ids=payload.document_ids,
        )
    except LlmUnavailable as exc:
        # Transient and already retried. 503 invites the client to try
        # again, which is the correct advice here and the wrong advice
        # for the branch below.
        logger.warning("answering unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answering model is unavailable right now. Please try again.",
        ) from exc
    except LlmError as exc:
        # A missing API key, a rejected request, a malformed schema. No
        # amount of retrying fixes any of them, so this is our bug and
        # 500 says so rather than inviting a retry loop.
        logger.exception("answering is misconfigured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answering is misconfigured on the server",
        ) from exc
    except ValueError as exc:
        # Retrieval's own validation. The request passed this layer's
        # schema, so disagreement means the embedder is misconfigured —
        # the server's fault, not the question's. Same mapping /search
        # already uses.
        logger.exception("retrieval rejected a question it should have accepted")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is misconfigured on the server",
        ) from exc

    message_id = None
    if conversation is not None:
        message_id = (await append_turn(db, conversation, answer)).id

    return AskResponse(
        answer=answer.text,
        question=answer.question,
        search_query=answer.search_query,
        cited_markers=[citation.marker for citation in answer.cited],
        grounded=answer.grounded,
        retrieval=RetrievalRead.from_context(answer.context),
        conversation_id=conversation.id if conversation is not None else None,
        message_id=message_id,
        dropped_sources=answer.dropped_sources,
        truncated=answer.truncated,
    )
