import pytest

from app.services.answering.answering_prompt import Turn
from app.services.answering.condense import (
    MAX_QUERY_CHARS,
    condense_question,
)
from app.services.llm.base import LlmUnavailable
from app.services.llm.fake_client import FakeChatClient

HISTORY = (
    Turn(role="user", content="what did I save about the coffee roastery?"),
    Turn(role="assistant", content="An invoice from Kyiv Coffee Roasters [1]."),
)


@pytest.mark.anyio
async def test_the_first_question_is_not_rewritten_and_costs_no_call():
    """
    The first question in a thread is standalone by definition. Skipping
    the call is most of what keeps a one-shot /ask as fast as it was
    before conversations existed — and a model call per question that
    needs none is a bill nobody notices until it arrives.
    """
    chat = FakeChatClient("should never be used")

    result = await condense_question(chat, "what is the total?", [])

    assert result == "what is the total?"
    assert chat.calls == [], "no history means no condensation call"


@pytest.mark.anyio
async def test_a_follow_up_is_rewritten_to_stand_alone():
    chat = FakeChatClient("when is the Kyiv Coffee Roasters invoice due?")

    result = await condense_question(chat, "when is it due?", HISTORY)

    assert result == "when is the Kyiv Coffee Roasters invoice due?"


@pytest.mark.anyio
async def test_the_conversation_reaches_the_rewriter():
    """
    A rewriter that cannot see the history can only echo the question
    back, which looks like success and retrieves nothing.
    """
    chat = FakeChatClient(lambda system, user: "rewritten" if "Roasters" in user else "")

    assert await condense_question(chat, "when is it due?", HISTORY) == "rewritten"


@pytest.mark.anyio
async def test_the_rewrite_is_deterministic():
    """
    A user who repeats a question gets the same search, or they see two
    different answers with no way to tell which is right.
    """
    chat = FakeChatClient("standalone question")
    await condense_question(chat, "when is it due?", HISTORY)

    assert chat.calls[0]["temperature"] == 0.0


@pytest.mark.anyio
async def test_a_model_failure_falls_back_instead_of_raising():
    """
    The condensation is an optimisation, not the answer. Losing it
    should cost retrieval quality, never the whole request — and if the
    model really is down, the generation call right after this will say
    so properly.
    """
    chat = FakeChatClient(error=LlmUnavailable("groq is down"))

    assert await condense_question(chat, "when is it due?", HISTORY) == "when is it due?"


@pytest.mark.anyio
async def test_a_model_that_answers_instead_of_rewriting_is_rejected():
    """
    The dangerous failure. A model that ignores the instruction produces
    something long and declarative; embedding that retrieves documents
    similar to an invented answer, which fails plausibly rather than
    visibly.
    """
    chat = FakeChatClient("Well, " + "the invoice is due in fourteen days. " * 20)

    result = await condense_question(chat, "when is it due?", HISTORY)

    assert result == "when is it due?"


@pytest.mark.anyio
async def test_an_empty_rewrite_falls_back():
    chat = FakeChatClient("   ")
    assert await condense_question(chat, "when is it due?", HISTORY) == "when is it due?"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "wrapped",
    [
        '"when is the invoice due?"',
        "'when is the invoice due?'",
        "  when is the invoice due?  ",
        "when is the invoice\n  due?",
    ],
)
async def test_quoting_and_whitespace_are_cleaned_off(wrapped):
    """
    Models wrap a rewrite in quotes despite being told not to. Left in,
    the quotes are embedded along with the question.
    """
    chat = FakeChatClient(wrapped)

    assert await condense_question(chat, "x", HISTORY) == "when is the invoice due?"


@pytest.mark.anyio
async def test_a_rewrite_at_the_limit_is_still_accepted():
    """
    The boundary. A cap that rejected everything near it would throw
    away good rewrites of genuinely long questions.
    """
    exact = "a" * MAX_QUERY_CHARS
    chat = FakeChatClient(exact)

    assert await condense_question(chat, "fallback", HISTORY) == exact

    over = FakeChatClient("a" * (MAX_QUERY_CHARS + 1))
    assert await condense_question(over, "fallback", HISTORY) == "fallback"
