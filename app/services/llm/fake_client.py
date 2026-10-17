from collections.abc import Callable

from app.services.llm.base import DEFAULT_MAX_TOKENS, Completion

FAKE_MODEL = "fake-chat-model"


class FakeChatClient:
    """
    Returns scripted responses and records every request.

    `responses` takes one of three forms, in rising order of control:

        "a string"        the same answer every time
        ["first", ...]    one per call, in order
        lambda s, u: ...  computed from the prompt

    The callable form is what makes an answering test realistic: a fake
    that returns "[1] yes" regardless of input cannot tell a correct
    prompt from an empty one, so the test passes even when the sources
    never reached the model.
    """

    def __init__(
        self,
        responses: str | list[str] | Callable[[str, str], str] = "I do not know.",
        *,
        finish_reason: str = "stop",
        error: Exception | None = None,
    ) -> None:
        self._responses = responses
        self._finish_reason = finish_reason
        self._error = error
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return FAKE_MODEL

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )

        if self._error is not None:
            raise self._error

        if callable(self._responses):
            text = self._responses(system, user)
        elif isinstance(self._responses, str):
            text = self._responses
        else:
            # Past the end of a scripted list is a test that made more
            # calls than it meant to — a silent extra model call is
            # exactly the bug worth failing on.
            index = len(self.calls) - 1
            if index >= len(self._responses):
                raise AssertionError(
                    f"FakeChatClient was called {len(self.calls)} times but "
                    f"only {len(self._responses)} response(s) were scripted"
                )
            text = self._responses[index]

        return Completion(text=text, model=FAKE_MODEL, finish_reason=self._finish_reason)
