import re
from dataclasses import dataclass

from app.services.retrieval.citations import Citation

# A marker, together with the horizontal whitespace on BOTH sides of it.
#
# Both sides are captured so that removing a marker leaves the sentence
# spaced correctly without a cleanup pass guessing afterwards. "net 14
# days [4]." has to become "net 14 days." while "days [4] and" has to
# become "days and"
_MARKERS = re.compile(r"([ \t]*)\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]([ \t]*)")

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(frozen=True)
class GroundedAnswer:
    """
    `text` is what to show the user. `cited` is the footnote list that
    goes beside it — in retrieval order, not order of mention, so the
    strongest source stays at the top where it was ranked.
    """

    text: str
    cited: tuple[Citation, ...]
    uncited: tuple[Citation, ...]
    invented: tuple[str, ...]

    @property
    def is_grounded(self) -> bool:
        return bool(self.cited)


def used_markers(text: str) -> tuple[str, ...]:
    """
    Reads the comma form too, so "[1, 2]" counts as two references rather
    than none — which is the difference between an answer being recorded
    as ungrounded and being recorded correctly.
    """
    seen: list[str] = []
    for _, group, _trailing in _MARKERS.findall(text):
        for number in group.split(","):
            marker = f"[{number.strip()}]"
            if marker not in seen:
                seen.append(marker)
    return tuple(seen)


def ground_answer(text: str, citations: tuple[Citation, ...] | list[Citation]) -> GroundedAnswer:
    """
    Reconcile an answer's markers with the sources it was given.
    """
    known = {citation.marker: citation for citation in citations}
    invented: list[str] = []

    def rewrite(match: re.Match[str]) -> str:
        leading, group, trailing = match.group(1), match.group(2), match.group(3)
        kept: list[str] = []
        for number in group.split(","):
            marker = f"[{number.strip()}]"
            if marker in known:
                if marker not in kept:
                    kept.append(marker)
            elif marker not in invented:
                invented.append(marker)

        if kept:
            return leading + "".join(kept) + trailing

        # Nothing survived, so the marker and its spacing both go — but
        # the spacing has to come out the way it would have been had the
        # marker never been written.
        if leading and match.string[match.end() : match.end() + 1] == "[":
            return leading

        # Otherwise one space survives only between two words; next to
        # punctuation or at a line start, none does.
        return " " if leading and trailing else ""

    cleaned = _TRAILING_SPACE.sub("", _MARKERS.sub(rewrite, text)).strip()

    referenced = set(used_markers(cleaned))
    return GroundedAnswer(
        text=cleaned,
        # Retrieval order, filtered — not order of mention. The panel is
        # ranked by how well each source matched the question, and the
        # model mentioning the third one first does not change that.
        cited=tuple(c for c in citations if c.marker in referenced),
        uncited=tuple(c for c in citations if c.marker not in referenced),
        invented=tuple(invented),
    )


__all__ = ["GroundedAnswer", "ground_answer", "used_markers"]
