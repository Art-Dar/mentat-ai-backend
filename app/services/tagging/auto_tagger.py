import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Protocol

import groq
from groq import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_TAGS = 5
MAX_CHARS_FOR_TAGGING = 4000
# How many existing tags to offer as reuse candidates
VOCABULARY_LIMIT = 60
MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 3.0

# Transient. AuthenticationError and BadRequestError are absent on purpose:
# a wrong API key or a malformed schema will not fix itself, and retrying
# only delays an identical failure.
_RETRYABLE = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tags"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = f"""\
You tag documents for a personal knowledge base shared by two people.

Rules:
- Propose at most {MAX_TAGS} tags, fewer when the document is narrow.
- REUSE a tag from the existing vocabulary whenever it fits, spelled
  exactly as given. Only invent a tag when nothing existing applies.
- Tags are topics, not summaries: "Machine Learning", not "an article
  about how transformers work".
- Always write tags in English, even when the document is not.
- 1-3 words, in Title Case.
- confidence is 0.0-1.0: how sure you are the tag belongs on THIS
  document, not how important the topic is.
"""


class TaggingError(RuntimeError):
    """Tags could not be produced. Never fails the document."""


@dataclass(frozen=True)
class TagSuggestion:
    name: str
    confidence: float


class Tagger(Protocol):
    """What the pipeline needs. Implemented by Groq and by the test fake."""

    async def suggest_tags(
        self,
        *,
        title: str | None,
        text: str,
        vocabulary: list[str],
    ) -> list[TagSuggestion]: ...


class GroqTagger:
    """Tagger backed by the Groq API."""

    def __init__(
        self,
        client: groq.AsyncGroq | None = None,
        model: str | None = None,
        max_tags: int = MAX_TAGS,
    ) -> None:
        if client is None:
            if not settings.groq_api_key:
                raise TaggingError("GROQ_API_KEY is not set")
            client = groq.AsyncGroq(api_key=settings.groq_api_key)
        self._client = client
        self._model = model or settings.tagging_model
        self._max_tags = max_tags

    async def suggest_tags(
        self,
        *,
        title: str | None,
        text: str,
        vocabulary: list[str],
    ) -> list[TagSuggestion]:
        if not text.strip():
            return []

        response = await self._call(self._build_prompt(title, text, vocabulary))
        return self._parse(response)

    def _build_prompt(self, title: str | None, text: str, vocabulary: list[str]) -> str:
        excerpt = text[:MAX_CHARS_FOR_TAGGING]
        known = ", ".join(vocabulary[:VOCABULARY_LIMIT]) or "(none yet)"
        return f"Existing vocabulary: {known}\n\nTitle: {title or '(none)'}\n\nDocument:\n{excerpt}"

    async def _call(self, prompt: str):
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._client.chat.completions.create(
                    model=self._model,
                    max_completion_tokens=512,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    # strict=True is constrained decoding: the model is
                    # physically unable to emit anything but JSON matching
                    # this schema. Without it the schema is a suggestion and
                    # a 400 comes back when the model wanders.
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "document_tags",
                            "strict": True,
                            "schema": _RESPONSE_SCHEMA,
                        },
                    },
                )
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "groq %s (attempt %d/%d), retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # auth, bad request, anything else
                raise TaggingError(f"{type(exc).__name__}: {exc}") from exc

        raise TaggingError(
            f"Groq unavailable after {MAX_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _parse(self, response) -> list[TagSuggestion]:
        """
        Turn the response into suggestions, refusing anything malformed.

        The schema holds only when the model finishes normally. A response
        cut off at max_completion_tokens still parses as JSON often enough
        to be dangerous, so finish_reason is checked first.
        """
        if not response.choices:
            raise TaggingError("response contained no choices")

        choice = response.choices[0]
        if choice.finish_reason not in {"stop", None}:
            raise TaggingError(f"unusable response: finish_reason={choice.finish_reason}")

        content = choice.message.content
        if not content:
            raise TaggingError("response contained no content")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TaggingError(f"response was not JSON: {exc}") from exc

        suggestions = [
            TagSuggestion(name=str(item["name"]).strip(), confidence=float(item["confidence"]))
            for item in payload.get("tags", [])
            if str(item.get("name", "")).strip()
        ]

        # The model is asked for at most MAX_TAGS and usually obeys, but a
        # cap the database would not enforce belongs in code, not a prompt.
        return [
            TagSuggestion(s.name, min(max(s.confidence, 0.0), 1.0))
            for s in suggestions[: self._max_tags]
        ]


def get_tagger() -> Tagger:
    """The tagger the pipeline uses. Replace in tests by passing one in."""
    return GroqTagger()