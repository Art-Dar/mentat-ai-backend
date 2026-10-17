from dataclasses import dataclass

from app.services.retrieval.citations import SpanProvenance, span_provenance

# How much document to show either side of the passage.
WINDOW_CHARS = 600
# A ceiling, so this endpoint cannot be turned into a way to dump a
# 2 MB document through a query string one window at a time.
MAX_WINDOW_CHARS = 4000


class PassageUnavailable(Exception):
    """
    The span cannot be read out of this document's text.
    """


@dataclass(frozen=True)
class Passage:
    """
    A cited span, in context, ready to render.
    They are what the client wraps in a <mark>. The absolute offsets are
    carried alongside so the same request can be made again — to widen
    the window, say — without the client recomputing anything.
    """

    text: str
    highlight_start: int
    highlight_end: int

    start_char: int
    end_char: int

    # How much of the document lies outside the window, so a client can
    # decide whether to offer "show more" and in which direction.
    chars_before: int
    chars_after: int
    char_count: int

    provenance: SpanProvenance

    @property
    def highlight(self) -> str:
        """The cited passage itself, sliced out of the window."""
        return self.text[self.highlight_start : self.highlight_end]


def _snap_forward(text: str, index: int, limit: int) -> int:
    """
    Move the window's left edge forward off a partial word.
    """
    if index <= 0:
        return 0

    cursor = index
    while cursor < limit and not text[cursor].isspace():
        cursor += 1

    if cursor >= limit:
        return index

    while cursor < limit and text[cursor].isspace():
        cursor += 1
    return cursor


def _snap_back(text: str, index: int, floor: int) -> int:
    """Move the window's right edge back off a partial word."""
    if index >= len(text):
        return len(text)

    cursor = index
    while cursor > floor and not text[cursor - 1].isspace():
        cursor -= 1

    # Same asymmetry as _snap_forward: no whitespace to snap to means
    # keep the edge, and landing exactly on the span's end is success.
    if cursor <= floor:
        return index

    while cursor > floor and text[cursor - 1].isspace():
        cursor -= 1
    return cursor


def extract_passage(
    content: str | None,
    start_char: int,
    end_char: int,
    *,
    window: int = WINDOW_CHARS,
    page_map: list[dict] | None = None,
) -> Passage:
    """
    One cited span with `window` characters of document either side.
    """
    if not content:
        raise PassageUnavailable(
            "This document has no extracted text, so there is nothing to show."
        )

    length = len(content)
    if start_char < 0 or start_char >= length:
        raise PassageUnavailable(
            f"That passage starts at character {start_char:,}, but this "
            f"document is {length:,} characters long. The document was "
            "most likely re-processed after the citation was made."
        )

    end_char = min(end_char, length)
    if end_char <= start_char:
        raise PassageUnavailable(
            f"That passage is empty: it ends at character {end_char:,}, "
            f"at or before its start at {start_char:,}."
        )

    window = max(0, min(window, MAX_WINDOW_CHARS))

    # Snap only the edges. The span itself is never moved — the whole
    # point is that the highlight marks exactly what was cited.
    window_start = _snap_forward(content, max(0, start_char - window), start_char)
    window_end = _snap_back(content, min(length, end_char + window), end_char)

    return Passage(
        text=content[window_start:window_end],
        highlight_start=start_char - window_start,
        highlight_end=end_char - window_start,
        start_char=start_char,
        end_char=end_char,
        chars_before=window_start,
        chars_after=length - window_end,
        char_count=length,
        provenance=span_provenance(start_char, end_char, page_map),
    )


__all__ = [
    "MAX_WINDOW_CHARS",
    "WINDOW_CHARS",
    "Passage",
    "PassageUnavailable",
    "extract_passage",
]
