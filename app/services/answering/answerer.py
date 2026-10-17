import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.answering.answering_context import RetrievalContext, build_retrieval_context
from app.services.answering.answering_prompt import Turn, build_prompt
from app.services.answering.condense import condense_question
from app.services.answering.markers import ground_answer
from app.services.ingestion.embedder import Embedder
from app.services.llm.base import ChatClient
from app.services.retrieval.citations import Citation, build_citations
from app.services.retrieval.search import DEFAULT_TOP_K, search_by_text

logger = logging.getLogger(__name__)

# Room for a few paragraphs. The system prompt asks for two or three
# sentences, so this is a ceiling against a runaway generation rather
# than a target — but it has to clear the sentence the model is on when
# it reaches the limit, because an answer cut mid-citation loses the
# citation.
MAX_ANSWER_TOKENS = 800

# Low, not zero. Zero makes a model repeat a bad phrasing verbatim every
# time a user retries; this is still close to deterministic for the same
# question over the same sources, which is what "verify my answer" needs.
ANSWER_TEMPERATURE = 0.2


@dataclass(frozen=True)
class Answer:
    """
    Everything the HTTP layer and the conversation store need.

    `context` holds every citation the model was SHOWN, not only the ones
    it used — a source that was retrieved and ignored is still something
    the user may want to look at, and hiding it would misrepresent what
    the assistant consulted. `cited` says which ones the prose actually
    references, so a client can emphasise those and dim the rest.
    """

    text: str
    question: str
    # What was actually embedded. Differs from `question` only on a
    # follow-up, and is returned because it is model output: a silently
    # bad rewrite looks exactly like bad retrieval.
    search_query: str
    context: RetrievalContext
    cited: tuple[Citation, ...] = ()
    model: str = ""
    # The model wrote markers that resolved to nothing. Always empty in
    # the response text — they are stripped — but counted here, because a
    # rising number means the prompt is losing control of the model and
    # nothing else in the system would show it.
    invented_markers: tuple[str, ...] = ()
    # Sources that did not fit the context budget.
    dropped_sources: int = 0
    truncated: bool = False
    history_used: int = 0

    @property
    def grounded(self) -> bool:
        """
        Whether the answer references any source.

        False is not an error: "I could not find that in your documents"
        is the correct answer to an unanswerable question and cites
        nothing. It means the prose carries no reference, so a client
        should not present it as verifiable.
        """
        return bool(self.cited)


async def answer_question(
    db: AsyncSession,
    *,
    question: str,
    user_id: uuid.UUID,
    embedder: Embedder,
    chat: ChatClient,
    history: tuple[Turn, ...] | list[Turn] = (),
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = 0.0,
    document_ids: list[uuid.UUID] | None = None,
) -> Answer:
    """
    Answer from the caller's own documents, with references.

    Raises LlmError (and its subclasses) if generation fails, and
    ValueError if retrieval is misconfigured. Both are left to the HTTP
    layer: a service that swallowed them would have to invent a
    user-facing message, and the one place that knows what the user
    should see is the route.

    An empty knowledge base is NOT an error. The prompt still asks the
    question, the system rules still forbid answering from general
    knowledge, and the model says it does not know — which is the correct
    answer and keeps the caller from needing a special case.
    """
    history = tuple(history)

    search_query = await condense_question(chat, question, history)

    hits = await search_by_text(
        db,
        search_query,
        user_id=user_id,
        embedder=embedder,
        top_k=top_k,
        min_similarity=min_similarity,
        document_ids=document_ids,
    )

    citations = await build_citations(db, hits)
    prompt = build_prompt(question, citations, history=history)

    completion = await chat.complete(
        system=prompt.system,
        user=prompt.user,
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=ANSWER_TEMPERATURE,
    )

    if completion.truncated:
        # Returned anyway: a cut-off answer with working citations is
        # more use than a 500, and the user can see it is incomplete.
        # Logged because it means MAX_ANSWER_TOKENS is too low for the
        # questions people actually ask.
        logger.warning("answer hit the token limit for a %d-char question", len(question))

    grounded = ground_answer(completion.text, prompt.citations)
    if grounded.invented:
        logger.warning(
            "model cited %s, which %s not exist",
            ", ".join(grounded.invented),
            "does" if len(grounded.invented) == 1 else "do",
        )

    # Built from the prompt's citations, not the grounded subset: the
    # panel shows what was consulted, and `cited` distinguishes what was
    # used. Passing only the used ones would make a source the assistant
    # looked at and dismissed invisible to the person checking its work.
    context = await build_retrieval_context(db, prompt.citations)

    return Answer(
        text=grounded.text,
        question=question,
        search_query=search_query,
        context=context,
        cited=grounded.cited,
        model=completion.model,
        invented_markers=grounded.invented,
        dropped_sources=len(prompt.dropped),
        truncated=completion.truncated,
        history_used=len(history),
    )


__all__ = ["ANSWER_TEMPERATURE", "MAX_ANSWER_TOKENS", "Answer", "answer_question"]
