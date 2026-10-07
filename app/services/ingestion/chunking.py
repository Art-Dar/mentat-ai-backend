import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150

_MIN_CHUNK_RATIO = 0.5
_WORD_SNAP_WINDOW = 50

# break candidates, best first
_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")
_WHITESPACE = re.compile(r"\s")


@dataclass(frozen=True)
class TextChunk:
    index: int # 0-based position within the document
    text: str  # [start_char:end_char]
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    length = len(text)
    if length == 0:
        return []

    chunks: list[TextChunk] = []
    pos = 0
    index = 0

    while pos < length:
        limit = min(pos + chunk_size, length)
        boundary = limit if limit >= length else _find_break(text, pos, limit, chunk_size)

        start, end = _trim(text, pos, boundary)
        if end > start:
            chunks.append(
                TextChunk(index=index, text=text[start:end], start_char=start, end_char=end)
            )
            index += 1

        if boundary >= length:
            break

        pos = _next_start(text, pos, boundary, overlap)

    return chunks


def _find_break(text: str, start: int, limit: int, chunk_size: int) -> int:

    earliest = min(start + int(chunk_size * _MIN_CHUNK_RATIO), limit)

    for separator in _SEPARATORS:
        found = text.rfind(separator, earliest, limit)
        if found != -1:
            return found + len(separator)

    return limit


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _next_start(text: str, pos: int, boundary: int, overlap: int) -> int:
    candidate = boundary - overlap

    if candidate > 0 and not text[candidate - 1].isspace():
        window_start = max(pos + 1, candidate - _WORD_SNAP_WINDOW)
        snapped = _rfind_whitespace(text, window_start, candidate)
        if snapped is not None:
            candidate = snapped + 1

    return max(candidate, pos + 1)


def _rfind_whitespace(text: str, start: int, end: int) -> int | None:
    for i in range(end - 1, start - 1, -1):
        if _WHITESPACE.match(text[i]):
            return i
    return None