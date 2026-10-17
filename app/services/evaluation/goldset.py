import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass, field


class GoldSetError(ValueError):
    """The gold set file is malformed."""


def normalize(text: str) -> str:
    """
    Whitespace is collapsed because chunk text keeps the document's line
    breaks and nobody writing a label reproduces them. Case is folded and
    Unicode is NFKC-normalised so that a label typed with a plain hyphen
    matches a PDF's en-dash, which is the single most common reason a
    correct label scores zero.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


@dataclass(frozen=True)
class GoldQuestion:
    """
    One labelled question.
    """

    question: str
    answer_spans: tuple[str, ...]
    # Free-text, for slicing results later — "ocr", "multi-page",
    # "needs-two-sources". Tuning that improves the average while
    # destroying one category is a common and invisible outcome.
    tags: tuple[str, ...] = ()
    note: str = ""
    _normalized: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def matches(self, chunk_text: str) -> bool:
        """Whether this chunk contains any of the answer spans."""
        haystack = normalize(chunk_text)
        return any(span in haystack for span in self.normalized_spans)

    @property
    def normalized_spans(self) -> tuple[str, ...]:
        return self._normalized or tuple(normalize(s) for s in self.answer_spans)


def load_goldset(path: str | pathlib.Path) -> tuple[GoldQuestion, ...]:
    """
    Validation is strict and loud. A silently skipped malformed entry
    lowers recall for a reason that has nothing to do with retrieval, and
    the resulting number looks exactly like a real regression.
    """
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldSetError(f"no gold set at {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldSetError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise GoldSetError(f"{path} should hold a list of questions")

    questions: list[GoldQuestion] = []
    for position, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            raise GoldSetError(f"entry {position} is not an object")

        question = str(entry.get("question", "")).strip()
        if not question:
            raise GoldSetError(f"entry {position} has no question")

        raw_spans = entry.get("answer_spans") or []
        if isinstance(raw_spans, str):
            raw_spans = [raw_spans]
        spans = tuple(str(s).strip() for s in raw_spans if str(s).strip())
        if not spans:
            raise GoldSetError(
                f"entry {position} ({question[:40]!r}) has no answer_spans — "
                "a question with nothing to match against cannot be scored, "
                "and counting it as a miss would understate every setting"
            )

        normalized = tuple(normalize(s) for s in spans)
        for span, folded in zip(spans, normalized, strict=True):
            # A span of one or two words matches somewhere in almost any
            # chunk, so it scores a hit for a retrieval that found
            # nothing relevant. Caught here rather than quietly inflating
            # recall for every setting equally — which is worse than a
            # wrong number, because it is a wrong number that looks
            # stable.
            if len(folded) < 12:
                raise GoldSetError(
                    f"entry {position}: answer span {span!r} is too short to "
                    "identify a passage; use a distinctive phrase"
                )

        questions.append(
            GoldQuestion(
                question=question,
                answer_spans=spans,
                tags=tuple(entry.get("tags") or ()),
                note=str(entry.get("note", "")),
                _normalized=normalized,
            )
        )

    if not questions:
        raise GoldSetError(f"{path} is empty")
    return tuple(questions)


def check_spans_exist(questions: tuple[GoldQuestion, ...], corpus: dict[str, str]) -> list[str]:
    """
    Which labels cannot possibly be matched by any document in the corpus.
    """
    haystacks = {name: normalize(text) for name, text in corpus.items()}
    unmatched: list[str] = []

    for question in questions:
        if not any(
            span in haystack
            for span in question.normalized_spans
            for haystack in haystacks.values()
        ):
            unmatched.append(
                f"{question.question!r}: none of {list(question.answer_spans)} "
                "appear in any document"
            )
    return unmatched


__all__ = [
    "GoldQuestion",
    "GoldSetError",
    "check_spans_exist",
    "load_goldset",
    "normalize",
]
