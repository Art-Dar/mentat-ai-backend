import json
import logging
from dataclasses import dataclass
from typing import Protocol

import groq

from app.services.llm.base import ChatClient, Completion, LlmError
from app.services.llm.groq_client import GroqChatClient

logger = logging.getLogger(__name__)

MAX_TAGS = 5
MAX_CHARS_FOR_TAGGING = 4000
# How many existing tags to offer as reuse candidates
VOCABULARY_LIMIT = 60
MAX_COMPLETION_TOKENS = 512

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
        client: groq.AsyncGroq | ChatClient | None = None,
        model: str | None = None,
        max_tags: int = MAX_TAGS,
    ) -> None:
        """
        `client` still accepts a raw groq.AsyncGroq, which is what the
        existing tests inject, so the stubs that record a request keep
        working unchanged
        """
        try:
            self._chat: ChatClient = (
                client if hasattr(client, "complete") else GroqChatClient(client, model)  # type: ignore[arg-type]
            )
        except LlmError as exc:
            # A missing API key surfaces as the tagging layer's own error,
            # because the pipeline catches TaggingError and lets the
            # document through untagged
            raise TaggingError(str(exc)) from exc
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

        try:
            completion = await self._chat.complete(
                system=_SYSTEM_PROMPT,
                user=self._build_prompt(title, text, vocabulary),
                max_tokens=MAX_COMPLETION_TOKENS,
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
        except LlmError as exc:
            raise TaggingError(str(exc)) from exc

        return self._parse(completion)

    def _build_prompt(self, title: str | None, text: str, vocabulary: list[str]) -> str:
        excerpt = text[:MAX_CHARS_FOR_TAGGING]
        known = ", ".join(vocabulary[:VOCABULARY_LIMIT]) or "(none yet)"
        return f"Existing vocabulary: {known}\n\nTitle: {title or '(none)'}\n\nDocument:\n{excerpt}"

    def _parse(self, completion: Completion) -> list[TagSuggestion]:
        """
        Turn the response into suggestions, refusing anything malformed.

        The schema holds only when the model finishes normally. A response
        cut off at max_completion_tokens still parses as JSON often enough
        to be dangerous, so finish_reason is checked first.
        """
        if completion.finish_reason not in {"stop", None}:
            raise TaggingError(f"unusable response: finish_reason={completion.finish_reason}")

        try:
            payload = json.loads(completion.text)
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
