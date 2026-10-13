"""
A deterministic Tagger for tests and for running the pipeline without an
Anthropic API key.
"""

import re

from app.services.tagging.auto_tagger import MAX_TAGS, TagSuggestion

_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


class FakeTagger:
    """
    Proposes the longest distinct words in the text as tags.
    Deterministic, free, and offline. The tags are not meaningful
    """

    def __init__(self, max_tags: int = MAX_TAGS) -> None:
        self.max_tags = max_tags
        # (title, number of vocabulary entries offered) per call.
        self.calls: list[tuple[str | None, int]] = []

    async def suggest_tags(
        self,
        *,
        title: str | None,
        text: str,
        vocabulary: list[str],
    ) -> list[TagSuggestion]:
        self.calls.append((title, len(vocabulary)))
        if not text.strip():
            return []

        # Reuse an existing tag when one appears verbatim in the text, so
        # the "reuse beats invention" behaviour is exercisable offline.
        lowered = text.lower()
        reused = [name for name in vocabulary if name.lower() in lowered]

        words = sorted({match.group().title() for match in _WORD.finditer(text)}, key=len)
        invented = [word for word in reversed(words) if word not in reused]

        chosen = (reused + invented)[: self.max_tags]
        return [
            TagSuggestion(name=name, confidence=round(0.9 - 0.1 * index, 2))
            for index, name in enumerate(chosen)
        ]
