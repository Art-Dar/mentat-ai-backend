import pytest

from app.services.ingestion.chunking import DEFAULT_CHUNK_SIZE, chunk_text

UK_PARAGRAPH = (
    "Сьогодні в Києві відбулася важлива зустріч представників різних галузей "
    "економіки для обговорення подальшого розвитку інфраструктури міста. "
)
EN_PARAGRAPH = (
    "Representatives from several industries met in the city center today to "
    "discuss further plans for economic development and infrastructure. "
)


def _long_text(paragraphs: int = 20) -> str:
    return "\n\n".join((UK_PARAGRAPH + EN_PARAGRAPH) for _ in range(paragraphs))


# the offset contract

def test_offsets_slice_back_to_the_chunk_text():
    text = _long_text()
    for chunk in chunk_text(text):
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_offsets_are_within_the_document():
    text = _long_text()
    for chunk in chunk_text(text):
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)


def test_indexes_are_sequential_from_zero():
    chunks = chunk_text(_long_text())
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_are_ordered_by_position():
    chunks = chunk_text(_long_text())
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)


# sizing and overlap

def test_no_chunk_exceeds_the_size_limit():
    for chunk in chunk_text(_long_text(), chunk_size=400, overlap=80):
        assert len(chunk.text) <= 400


def test_consecutive_chunks_overlap():
    chunks = chunk_text(_long_text(), chunk_size=400, overlap=120)
    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start_char < previous.end_char


def test_whole_document_is_covered():
    text = _long_text()
    chunks = chunk_text(text, chunk_size=400, overlap=100)

    covered = bytearray(len(text))
    for chunk in chunks:
        for i in range(chunk.start_char, chunk.end_char):
            covered[i] = 1

    # every non-whitespace character ends up in at least one chunk
    missed = [i for i, seen in enumerate(covered) if not seen and not text[i].isspace()]
    assert missed == []


# boundary preference

def test_prefers_paragraph_boundaries():
    text = "\n\n".join("Абзац номер %d з достатньою кількістю тексту." % i for i in range(12))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    # a chunk that broke at a paragraph ends with a full sentence, not mid-word
    assert all(chunk.text.endswith(".") or chunk is chunks[-1] for chunk in chunks)


def test_does_not_split_mid_word_when_a_space_is_available():
    text = " ".join(f"слово{i}" for i in range(300))
    for chunk in chunk_text(text, chunk_size=200, overlap=40):
        assert not chunk.text.startswith(" ")
        assert not chunk.text.endswith(" ")


# edge cases

def test_short_text_is_one_chunk():
    chunks = chunk_text("Короткий текст без розривів.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0


def test_empty_text_returns_nothing():
    assert chunk_text("") == []


def test_whitespace_only_text_returns_nothing():
    assert chunk_text("   \n\n\t  ") == []


def test_text_with_no_separators_is_hard_split():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_a_single_enormous_token_terminates():
    # a base64 blob pasted into a note — no whitespace anywhere
    text = "Q" * 50_000
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)


def test_text_exactly_at_the_limit_is_one_chunk():
    text = "x" * DEFAULT_CHUNK_SIZE
    assert len(chunk_text(text)) == 1


def test_chunks_never_contain_only_whitespace():
    text = "Перший абзац." + "\n" * 200 + "Другий абзац."
    for chunk in chunk_text(text, chunk_size=50, overlap=10):
        assert chunk.text.strip()


# argument validation

def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=100)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("text", overlap=-1)