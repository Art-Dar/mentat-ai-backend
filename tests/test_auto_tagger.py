import json

import groq
import httpx
import pytest
from pydantic import SecretStr

from app.services.tagging.auto_tagger import (
    MAX_CHARS_FOR_TAGGING,
    VOCABULARY_LIMIT,
    GroqTagger,
    TaggingError,
)
from app.services.tagging.fake_tagger import FakeTagger

ARTICLE = (
    "Retrieval augmented generation combines a vector search over your own "
    "documents with a language model that answers from what it retrieves."
)


class StubMessage:
    def __init__(self, content: str | None):
        self.content = content


class StubChoice:
    def __init__(self, content: str | None, finish_reason: str):
        self.message = StubMessage(content)
        self.finish_reason = finish_reason


class StubResponse:
    def __init__(self, choices):
        self.choices = choices


class StubCompletions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        return self._owner.response


class StubClient:
    """Records every request and returns a canned, schema-shaped reply."""

    def __init__(
        self,
        tags=None,
        finish_reason: str = "stop",
        content: str | None = None,
        choices=None,
    ):
        if choices is None:
            payload = {"tags": tags if tags is not None else [{"name": "RAG", "confidence": 0.9}]}
            body = content if content is not None else json.dumps(payload)
            choices = [StubChoice(body, finish_reason)]
        self.response = StubResponse(choices)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = StubCompletions(self)


class FailingClient:
    def __init__(self, error: Exception, failures: int):
        self.error = error
        self.failures = failures
        self.attempts = 0
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        payload = {"tags": [{"name": "RAG", "confidence": 0.9}]}
        return StubResponse([StubChoice(json.dumps(payload), "stop")])


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, request=request)


def rate_limited() -> Exception:
    return groq.RateLimitError("slow down", response=_response(429), body=None)


def bad_key() -> Exception:
    return groq.AuthenticationError("bad key", response=_response(401), body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    # The backoff moved into the shared LLM client, so the patch follows it.
    monkeypatch.setattr("app.services.llm.groq_client.asyncio.sleep", instant)


def tagger(client, **kwargs) -> GroqTagger:
    return GroqTagger(client=client, model="openai/gpt-oss-20b", **kwargs)


# the request


@pytest.mark.anyio
async def test_the_existing_vocabulary_is_offered_for_reuse():
    """
    The single most important property of an auto-tagger. Without it the
    same idea arrives as "ML", "machine-learning" and "Machine Learning"
    across three documents and every filter returns a third of the matches.
    """
    client = StubClient()

    await tagger(client).suggest_tags(
        title="RAG explained", text=ARTICLE, vocabulary=["Machine Learning", "Databases"]
    )

    prompt = client.calls[0]["messages"][1]["content"]
    assert "Machine Learning" in prompt
    assert "Databases" in prompt


@pytest.mark.anyio
async def test_long_documents_are_truncated():
    """
    Topic is obvious from the opening. On a free tier this is also the
    difference between staying inside tokens-per-minute and not.
    """
    client = StubClient()
    huge = "word " * 50_000

    await tagger(client).suggest_tags(title=None, text=huge, vocabulary=[])

    prompt = client.calls[0]["messages"][1]["content"]
    assert len(prompt) < MAX_CHARS_FOR_TAGGING + 500  # + the framing text


@pytest.mark.anyio
async def test_the_vocabulary_offered_is_capped():
    client = StubClient()
    enormous = [f"Tag{i}" for i in range(500)]

    await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=enormous)

    prompt = client.calls[0]["messages"][1]["content"]
    assert f"Tag{VOCABULARY_LIMIT - 1}" in prompt
    assert f"Tag{VOCABULARY_LIMIT}" not in prompt


@pytest.mark.anyio
async def test_strict_structured_output_is_requested():
    """
    strict=True is constrained decoding: the model cannot emit anything but
    JSON matching the schema. Without it the schema is a suggestion, and a
    400 comes back whenever the model wanders.
    """
    client = StubClient()

    await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    response_format = client.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert "tags" in response_format["json_schema"]["schema"]["properties"]


@pytest.mark.anyio
async def test_the_rules_are_sent_as_a_system_message():
    client = StubClient()

    await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    system = client.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "English" in system["content"]


@pytest.mark.anyio
async def test_empty_text_makes_no_api_call():
    client = StubClient()

    assert await tagger(client).suggest_tags(title=None, text="   ", vocabulary=[]) == []
    assert client.calls == []


# the response


@pytest.mark.anyio
async def test_suggestions_are_parsed():
    client = StubClient(
        tags=[
            {"name": "Retrieval Augmented Generation", "confidence": 0.93},
            {"name": "Vector Search", "confidence": 0.71},
        ]
    )

    suggestions = await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert [s.name for s in suggestions] == ["Retrieval Augmented Generation", "Vector Search"]
    assert suggestions[0].confidence == pytest.approx(0.93)


@pytest.mark.anyio
async def test_too_many_tags_are_capped():
    """A limit the database would not enforce belongs in code, not a prompt."""
    client = StubClient(tags=[{"name": f"Tag {i}", "confidence": 0.5} for i in range(20)])

    suggestions = await tagger(client, max_tags=3).suggest_tags(
        title=None, text=ARTICLE, vocabulary=[]
    )

    assert len(suggestions) == 3


@pytest.mark.anyio
async def test_blank_names_are_dropped():
    client = StubClient(
        tags=[{"name": "  ", "confidence": 0.9}, {"name": "Real Tag", "confidence": 0.8}]
    )

    suggestions = await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert [s.name for s in suggestions] == ["Real Tag"]


@pytest.mark.anyio
async def test_confidence_is_clamped_to_the_valid_range():
    """attach_tags rejects anything outside 0-1, so it is fixed here first."""
    client = StubClient(
        tags=[{"name": "Over", "confidence": 1.4}, {"name": "Under", "confidence": -0.2}]
    )

    suggestions = await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert [s.confidence for s in suggestions] == [1.0, 0.0]


@pytest.mark.anyio
async def test_a_truncated_response_is_refused():
    """
    The schema holds only when the model finishes normally. A reply cut off
    at max_completion_tokens parses as JSON often enough to be dangerous.
    """
    client = StubClient(finish_reason="length")

    with pytest.raises(TaggingError, match="length"):
        await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])


@pytest.mark.anyio
async def test_a_filtered_response_is_refused():
    client = StubClient(finish_reason="content_filter")

    with pytest.raises(TaggingError, match="content_filter"):
        await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])


@pytest.mark.anyio
async def test_non_json_is_reported_clearly():
    client = StubClient(content="I'm afraid I can't do that")

    with pytest.raises(TaggingError, match="not JSON"):
        await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])


@pytest.mark.anyio
async def test_an_empty_response_is_reported_clearly():
    client = StubClient(choices=[])

    with pytest.raises(TaggingError, match="no choices"):
        await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])


# failure policy


@pytest.mark.anyio
async def test_rate_limits_are_retried():
    """The free tier's defining failure mode: 30 requests a minute."""
    client = FailingClient(rate_limited(), failures=2)

    suggestions = await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert client.attempts == 3
    assert suggestions


@pytest.mark.anyio
async def test_a_bad_api_key_is_not_retried():
    """It will not fix itself; retrying only delays an identical failure."""
    client = FailingClient(bad_key(), failures=99)

    with pytest.raises(TaggingError, match="AuthenticationError"):
        await tagger(client).suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert client.attempts == 1


@pytest.mark.anyio
async def test_missing_api_key_names_the_setting(monkeypatch):
    """
    The key check moved into the shared LLM client, so the patch target
    follows it. What this pins is unchanged and is the part that matters:
    a missing key still surfaces as TaggingError, not as LlmError. The
    ingestion pipeline catches TaggingError and lets the document through
    untagged — leaking the client's own exception type would turn a
    missing key into a failed upload.
    """
    monkeypatch.setattr("app.services.llm.groq_client.settings.groq_api_key", SecretStr(""))

    with pytest.raises(TaggingError, match="GROQ_API_KEY"):
        GroqTagger()


# the fake used by pipeline tests


@pytest.mark.anyio
async def test_fake_tagger_is_deterministic():
    fake = FakeTagger()

    first = await fake.suggest_tags(title=None, text=ARTICLE, vocabulary=[])
    second = await fake.suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert first == second


@pytest.mark.anyio
async def test_fake_tagger_prefers_an_existing_tag():
    fake = FakeTagger()

    suggestions = await fake.suggest_tags(
        title=None, text="a note about vector search and nothing else", vocabulary=["Vector"]
    )

    assert suggestions[0].name == "Vector"


@pytest.mark.anyio
async def test_fake_tagger_respects_the_cap():
    fake = FakeTagger(max_tags=2)

    suggestions = await fake.suggest_tags(title=None, text=ARTICLE, vocabulary=[])

    assert len(suggestions) == 2
