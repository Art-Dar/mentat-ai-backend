import asyncio
import logging

import groq
from groq import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings
from app.services.llm.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    LlmError,
    LlmInvalidResponse,
    LlmUnavailable,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 25.0

_RETRYABLE = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)


class GroqChatClient:
    """
    Accepts an injected SDK client so a test can pass a stub that records
    the request — which is how the tagging suite has always worked, and
    why that suite needs no changes beyond where it patches sleep.
    """

    def __init__(
        self,
        client: groq.AsyncGroq | None = None,
        model: str | None = None,
        *,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if client is None:
            key = settings.groq_api_key.get_secret_value()
            if not key:
                # Not LlmUnavailable: no amount of retrying supplies a
                # missing API key, and the caller should surface this as
                # a deployment fault rather than "try again".
                raise LlmError("GROQ_API_KEY is not set")
            client = groq.AsyncGroq(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._client = client
        self._model = model or settings.tagging_model
        self._max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> Completion:
        request: dict = {
            "model": self._model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            request["response_format"] = response_format
        if temperature is not None:
            request["temperature"] = temperature

        return self._read(await self._call(request))

    async def _call(self, request: dict):
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self._client.chat.completions.create(**request)
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break
                delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "groq %s (attempt %d/%d), retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # auth, bad request, anything else
                raise LlmError(f"{type(exc).__name__}: {exc}") from exc

        raise LlmUnavailable(
            f"Groq unavailable after {self._max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _read(self, response) -> Completion:
        """
        Pull the one string out, refusing anything structurally wrong.
        """
        if not response.choices:
            raise LlmInvalidResponse("response contained no choices")

        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise LlmInvalidResponse("response contained no content")

        return Completion(
            text=content,
            model=getattr(response, "model", self._model),
            finish_reason=choice.finish_reason,
        )


def get_chat_client() -> GroqChatClient:
    """
    The client the API uses. Overridden in tests through FastAPI's
    dependency_overrides, the same way get_embedder and get_file_store
    are, so no test ever reaches Groq or spends a token.
    """
    return GroqChatClient()
