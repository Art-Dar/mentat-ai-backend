from dataclasses import dataclass
from typing import Protocol

# Enough for a few paragraphs. The answering system prompt asks for two
# or three sentences, so this is a ceiling against a runaway generation
# rather than a target.
DEFAULT_MAX_TOKENS = 1024


class LlmError(RuntimeError):
    """The model could not produce a usable response."""


class LlmUnavailable(LlmError):
    """
    Rate limits, timeouts, connection failures, provider 5xx. A later
    request may well succeed, which is what separates this from the rest
    — it is the difference between telling a user "try again in a moment"
    and telling them the server is misconfigured.
    """


class LlmInvalidResponse(LlmError):
    """
    The call succeeded and the result is unusable: no choices, empty
    content, or output that does not parse.
    """


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    finish_reason: str | None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class ChatClient(Protocol):
    """Implemented by GroqChatClient and by the test fake."""

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> Completion: ...


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "ChatClient",
    "Completion",
    "LlmError",
    "LlmInvalidResponse",
    "LlmUnavailable",
]
