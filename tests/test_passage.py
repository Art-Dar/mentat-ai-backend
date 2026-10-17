import uuid

import pytest

from app.models import Document, DocumentSource
from app.services.retrieval.citations import build_citation
from app.services.retrieval.passage import (
    MAX_WINDOW_CHARS,
    PassageUnavailable,
    extract_passage,
)
from app.services.retrieval.search import SearchHit

# Word lengths vary on purpose: a fixed-width filler would hide every
# word-boundary bug in here.
TEXT = (
    "Invoice 2026-0042 issued by Kyiv Coffee Roasters on 3 March. "
    "Payment terms: net 14 days from the invoice date. "
    "Total: 480.00 UAH including VAT at 20 percent. "
    "Remittance to IBAN UA213223130000026007233566001 reference 0042. "
    "Queries to accounts@example.com within seven days of receipt."
)

PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 110, "needs_ocr": False, "source": "pdf"},
    {
        "index": 1,
        "start_char": 110,
        "end_char": len(TEXT),
        "needs_ocr": False,
        "source": "ocr",
        "mean_confidence": 58.0,
    },
]


def span_of(needle: str) -> tuple[int, int]:
    start = TEXT.index(needle)
    return start, start + len(needle)


def test_the_highlight_slices_to_exactly_the_cited_span():
    """
    The one assertion this module exists for. highlight_start/end index
    the returned window, not the document, and getting that translation
    wrong shows the user the wrong sentence as evidence.
    """
    start, end = span_of("Total: 480.00 UAH")
    passage = extract_passage(TEXT, start, end)

    assert passage.highlight == "Total: 480.00 UAH"
    assert passage.text[passage.highlight_start : passage.highlight_end] == TEXT[start:end]


def test_context_arrives_on_both_sides():
    start, end = span_of("Total: 480.00 UAH")
    passage = extract_passage(TEXT, start, end, window=60)

    assert passage.highlight_start > 0, "nothing before the passage"
    assert passage.highlight_end < len(passage.text), "nothing after the passage"
    assert "net 14 days" in passage.text, "the preceding sentence should be visible"


def test_window_zero_returns_the_span_alone():
    start, end = span_of("net 14 days")
    passage = extract_passage(TEXT, start, end, window=0)

    assert passage.text == "net 14 days"
    assert (passage.highlight_start, passage.highlight_end) == (0, 11)


def test_the_window_never_opens_mid_word():
    """
    A pane that starts "…ment terms: net 14" reads as a rendering bug, and
    a user who thinks the evidence pane is broken will not trust the
    citation above it.
    """
    start, end = span_of("480.00")
    for window in range(1, 80):
        text = extract_passage(TEXT, start, end, window=window).text
        assert text == text.lstrip(), "leading whitespace left by the snap"
        first = text.split(" ", 1)[0]
        assert first in TEXT.split(), f"window {window} opened mid-word: {first!r}"


def test_a_single_enormous_token_is_not_swallowed():
    """
    The snap looks for whitespace and must give up rather than eat the
    span when there is none. An IBAN with no spaces around it is the real
    case this came from.
    """
    start, end = span_of("reference 0042")
    passage = extract_passage(TEXT, start, end, window=20)

    assert passage.highlight == "reference 0042"


def test_context_survives_a_token_longer_than_the_window():
    """
    The documented exception to "never opens mid-word", pinned.
    """
    # No space between the long run and the span: a chunk boundary falls
    # on a character count, so it lands mid-token routinely.
    text = "prefix " + "A" * 400 + "Total: 480.00 UAH due now"
    start = text.index("Total: 480.00 UAH")
    passage = extract_passage(text, start, start + 17, window=30)

    assert passage.highlight == "Total: 480.00 UAH"
    assert passage.highlight_start == 30, "the leading context was dropped"
    assert passage.text.startswith("A"), "expected the tail of the long token"


def test_a_span_at_the_very_start_does_not_run_off_the_front():
    passage = extract_passage(TEXT, 0, 17, window=500)

    assert passage.chars_before == 0
    assert passage.highlight_start == 0
    assert passage.highlight == "Invoice 2026-0042"


def test_a_span_at_the_very_end_reports_nothing_after_it():
    start = TEXT.index("within seven days")
    passage = extract_passage(TEXT, start, len(TEXT), window=500)

    assert passage.chars_after == 0
    assert passage.highlight_end == len(passage.text)


def test_chars_before_and_after_say_whether_to_offer_more():
    start, end = span_of("Total: 480.00 UAH")
    narrow = extract_passage(TEXT, start, end, window=10)
    whole = extract_passage(TEXT, start, end, window=MAX_WINDOW_CHARS)

    assert narrow.chars_before > 0 and narrow.chars_after > 0
    assert whole.chars_before == 0 and whole.chars_after == 0
    assert whole.char_count == len(TEXT)


def test_an_end_past_the_document_is_clamped_not_rejected():
    """
    A chunk legitimately ends at the last character, and an off-by-one in
    a page_map must not be the thing that stops someone checking a number
    against their own receipt.
    """
    passage = extract_passage(TEXT, len(TEXT) - 10, len(TEXT) + 5_000)

    assert passage.end_char == len(TEXT)
    assert passage.highlight == TEXT[-10:]


def test_an_oversized_window_is_capped_rather_than_honoured():
    start, end = span_of("net 14 days")
    passage = extract_passage(TEXT, start, end, window=MAX_WINDOW_CHARS * 100)

    assert len(passage.text) <= len(TEXT)


def test_a_start_past_the_end_is_refused_rather_than_guessed():
    """
    Means the document was re-ingested after the citation was built.
    Returning the tail of the document would present the wrong text as
    the cited evidence — silently, which is the whole problem.
    """
    with pytest.raises(PassageUnavailable) as caught:
        extract_passage(TEXT, len(TEXT) + 1, len(TEXT) + 50)

    message = str(caught.value)
    assert str(len(TEXT)) in message.replace(",", "")
    assert "re-processed" in message


def test_a_start_exactly_at_the_end_is_refused():
    """
    The boundary, which is where a stale offset most often lands: the
    document was shortened by a re-ingest and the old chunk now begins
    exactly where the text stops.
    """
    with pytest.raises(PassageUnavailable) as caught:
        extract_passage(TEXT, len(TEXT), len(TEXT) + 10)

    assert "re-processed" in str(caught.value)


@pytest.mark.parametrize("content", [None, ""])
def test_a_document_with_no_text_is_refused(content):
    """
    A document still ingesting, or one whose extraction failed. Either
    way there is nothing to verify, and the user should be told that
    rather than shown an empty pane.
    """
    with pytest.raises(PassageUnavailable):
        extract_passage(content, 0, 10)


def test_an_inverted_span_is_refused():
    with pytest.raises(PassageUnavailable):
        extract_passage(TEXT, 50, 50)


def test_a_negative_start_is_refused():
    with pytest.raises(PassageUnavailable):
        extract_passage(TEXT, -5, 10)


def test_the_pages_match_what_the_citation_said():
    """
    A footnote reading "page 2" beside a context pane headed "page 1"
    discredits both, and a user cannot tell which one lied. This asserts
    the two agree for every span across the page boundary — the region
    where any off-by-one would actually show up.
    """
    document = Document(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        source=DocumentSource.PDF,
        content=TEXT,
        title="Invoice 2026-0042",
        page_map=PAGE_MAP,
    )

    for start in range(90, 130):
        end = start + 40
        hit = SearchHit(
            chunk_id=uuid.uuid4(),
            document_id=document.id,
            chunk_index=0,
            content=TEXT[start:end],
            start_char=start,
            end_char=end,
            similarity=0.8,
            document_title=None,
            document_url=None,
        )
        citation = build_citation(hit, document, "[1]")
        passage = extract_passage(TEXT, start, end, page_map=PAGE_MAP)

        assert passage.provenance.pages == citation.pages, f"disagreed at {start}"
        assert passage.provenance.from_ocr == citation.from_ocr, f"disagreed at {start}"
        assert passage.provenance.ocr_confidence == citation.ocr_confidence
        assert passage.provenance.page_count == citation.page_count


def test_ocr_provenance_reaches_the_passage():
    """
    The context pane needs the same "read from a scan" warning the
    footnote carries, or a user reads the surrounding text as verified
    when it was guessed by an OCR engine at 58% confidence.
    """
    start = TEXT.index("Remittance")
    passage = extract_passage(TEXT, start, start + 20, page_map=PAGE_MAP)

    assert passage.provenance.from_ocr is True
    assert passage.provenance.ocr_confidence == 58.0


def test_a_document_without_pages_still_works():
    """Notes and articles have no page_map and are the main reason this
    endpoint exists at all — they have no file to open instead."""
    passage = extract_passage(TEXT, 10, 40, page_map=None)

    assert passage.provenance.pages == ()
    assert passage.provenance.page_count == 0
    assert passage.highlight == TEXT[10:40]
