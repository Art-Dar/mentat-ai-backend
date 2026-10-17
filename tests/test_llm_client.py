import groq
import httpx
import pytest
from pydantic import SecretStr

from app.services.llm.base import (
    Completion,
    LlmError,
    LlmInvalidResponse,
    LlmUnavailable,
)
from app.services.llm.fake_client import FakeChatClient
from app.services.llm.groq_client import MAX_ATTEMPTS, GroqChatClient


class StubMessage:
    def __init__(self, content):
        self.content = content


class StubChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = StubMessage(content)
        self.finish_reason = finish_reason


class StubResponse:
    def __init__(self, choices, model="openai/gpt-oss-20b"):
        self.choices = choices
        self.model = model


class StubClient:
    """Records every request and returns a canned reply."""

    def __init__(self, content="an answer", finish_reason="stop", choices=None):
        if choices is None:
            choices = [StubChoice(content, finish_reason)]
        self.response = StubResponse(choices)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FailingClient:
    def __init__(self, error, failures):
        self.error = error
        self.failures = failures
        self.attempts = 0
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return StubResponse([StubChoice("recovered")])


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, request=request)


def rate_limited():
    return groq.RateLimitError("slow down", response=_response(429), body=None)


def bad_key():
    return groq.AuthenticationError("bad key", response=_response(401), body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr("app.services.llm.groq_client.asyncio.sleep", instant)


def client(stub, **kwargs) -> GroqChatClient:
    return GroqChatClient(client=stub, model="openai/gpt-oss-20b", **kwargs)


@pytest.mark.anyio
async def test_system_and_user_become_two_messages():
    stub = StubClient()

    await client(stub).complete(system="be brief", user="what is the total?")

    messages = stub.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "be brief"
    assert messages[1]["content"] == "what is the total?"


@pytest.mark.anyio
async def test_response_format_is_sent_only_when_asked_for():
    """
    The tagger needs constrained JSON decoding; an answer must not be
    constrained to a schema. Sending the key unconditionally with a None
    value is a 400 from the provider.
    """
    plain = StubClient()
    await client(plain).complete(system="s", user="u")
    assert "response_format" not in plain.calls[0]

    structured = StubClient()
    await client(structured).complete(system="s", user="u", response_format={"type": "json_object"})
    assert structured.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.anyio
async def test_temperature_is_sent_only_when_set():
    stub = StubClient()
    await client(stub).complete(system="s", user="u")
    assert "temperature" not in stub.calls[0]

    stub = StubClient()
    await client(stub).complete(system="s", user="u", temperature=0.0)
    assert stub.calls[0]["temperature"] == 0.0


@pytest.mark.anyio
async def test_the_token_cap_is_passed_through():
    stub = StubClient()
    await client(stub).complete(system="s", user="u", max_tokens=99)
    assert stub.calls[0]["max_completion_tokens"] == 99


@pytest.mark.anyio
async def test_a_rate_limit_is_retried_and_can_succeed():
    stub = FailingClient(rate_limited(), failures=2)

    completion = await client(stub).complete(system="s", user="u")

    assert completion.text == "recovered"
    assert stub.attempts == 3


@pytest.mark.anyio
async def test_giving_up_is_unavailable_not_a_generic_error():
    """
    The distinction the route depends on. LlmUnavailable becomes a 503
    that invites a retry; anything else becomes a 500 that does not.
    """
    stub = FailingClient(rate_limited(), failures=MAX_ATTEMPTS + 5)

    with pytest.raises(LlmUnavailable):
        await client(stub).complete(system="s", user="u")

    assert stub.attempts == MAX_ATTEMPTS


@pytest.mark.anyio
async def test_a_bad_api_key_is_not_retried():
    """
    A wrong key will be just as wrong nine seconds from now. Retrying it
    spends the user's patience to reach the same failure.
    """
    stub = FailingClient(bad_key(), failures=99)

    with pytest.raises(LlmError) as caught:
        await client(stub).complete(system="s", user="u")

    assert not isinstance(caught.value, LlmUnavailable), "auth is not transient"
    assert stub.attempts == 1


def test_a_missing_api_key_fails_loudly_at_construction(monkeypatch):
    """
    Patched rather than relying on the environment: the test .env carries
    a placeholder key, so a version of this that just called
    GroqChatClient() passed or failed depending on who ran it.

    LlmError and NOT LlmUnavailable, because the route turns the latter
    into a 503 that tells the user to try again — and no amount of
    trying again supplies a missing key.
    """
    monkeypatch.setattr("app.services.llm.groq_client.settings.groq_api_key", SecretStr(""))

    with pytest.raises(LlmError, match="GROQ_API_KEY") as caught:
        GroqChatClient()

    assert not isinstance(caught.value, LlmUnavailable)


@pytest.mark.anyio
async def test_a_response_with_no_choices_is_invalid():
    with pytest.raises(LlmInvalidResponse):
        await client(StubClient(choices=[])).complete(system="s", user="u")


@pytest.mark.anyio
async def test_empty_content_is_invalid():
    with pytest.raises(LlmInvalidResponse):
        await client(StubClient(content=None)).complete(system="s", user="u")


@pytest.mark.anyio
async def test_finish_reason_is_reported_not_judged():
    """
    The caller decides what a truncated response means. For the tagger's
    JSON it is fatal — half a JSON object parses often enough to be
    dangerous. For prose it is not: a cut-off answer with working
    citations still beats a 500.
    """
    stub = StubClient(content="half an ans", finish_reason="length")

    completion = await client(stub).complete(system="s", user="u")

    assert completion.text == "half an ans"
    assert completion.truncated is True


def test_a_normal_finish_is_not_truncated():
    assert Completion(text="x", model="m", finish_reason="stop").truncated is False


@pytest.mark.anyio
async def test_the_fake_can_compute_its_answer_from_the_prompt():
    """
    The form that makes an answering test mean something. A fake that
    returns the same string regardless of input cannot tell a prompt
    carrying the sources from an empty one, so the test passes even when
    retrieval never reached the model.
    """
    fake = FakeChatClient(lambda system, user: "[1]" if "SOURCE" in user else "no idea")

    with_sources = await fake.complete(system="s", user="<<<SOURCE [1]>>> total 480")
    without = await fake.complete(system="s", user="Sources:\n(none)")

    assert with_sources.text == "[1]"
    assert without.text == "no idea"


@pytest.mark.anyio
async def test_the_fake_scripts_responses_in_order():
    fake = FakeChatClient(["first", "second"])

    assert (await fake.complete(system="s", user="u")).text == "first"
    assert (await fake.complete(system="s", user="u")).text == "second"


@pytest.mark.anyio
async def test_the_fake_refuses_an_unscripted_extra_call():
    """
    An unexpected extra model call is a real bug — a condensation that
    should have been skipped, a retry that should not have happened —
    and it costs money in production. Silently returning the last
    response would hide it.
    """
    fake = FakeChatClient(["only one"])
    await fake.complete(system="s", user="u")

    with pytest.raises(AssertionError, match="2 times"):
        await fake.complete(system="s", user="u")


@pytest.mark.anyio
async def test_the_fake_can_raise():
    fake = FakeChatClient(error=LlmUnavailable("groq is down"))

    with pytest.raises(LlmUnavailable):
        await fake.complete(system="s", user="u")
