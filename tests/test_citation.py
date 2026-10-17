import uuid

import pytest

from app.models import DocumentSource
from app.services.retrieval.citations import (
    FRAGMENT_WORDS,
    QUOTE_CHARS,
    Citation,
    _pages_for_span,
    build_citation,
    render_sources,
)
from app.services.retrieval.search import SearchHit


class FakeDocument:
    def __init__(
        self,
        *,
        page_map=None,
        source=DocumentSource.PDF,
        title="Invoice",
        url=None,
        original_filename=None,
    ):
        self.page_map = page_map
        self.source = source
        self.title = title
        self.url = url
        self.original_filename = original_filename


def pdf_pages(*spans, ocr_pages=(), confidences=None):
    confidences = confidences or {}
    return [
        {
            "index": index,
            "start_char": start,
            "end_char": end,
            "needs_ocr": False,
            "source": "ocr" if index in ocr_pages else "pdf",
            **({"mean_confidence": confidences[index]} if index in confidences else {}),
        }
        for index, (start, end) in enumerate(spans)
    ]


def hit(start=0, end=100, content="Total 349.50 UAH payable within fourteen days"):
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        content=content,
        start_char=start,
        end_char=end,
        similarity=0.82,
        document_title="Invoice",
        document_url=None,
    )


def test_a_chunk_inside_one_page_cites_that_page():
    pages, count = _pages_for_span(100, 400, pdf_pages((0, 770), (772, 1542)))
    assert pages == (1,), "page_map is 0-based; a citation must not be"
    assert count == 2


def test_a_chunk_that_straddles_a_boundary_cites_both_pages():
    pages, _ = _pages_for_span(615, 1541, pdf_pages((0, 770), (772, 1542), (1544, 2314)))
    assert pages == (1, 2)


def test_a_chunk_spanning_three_pages_reports_the_whole_range():
    pages, _ = _pages_for_span(10, 2000, pdf_pages((0, 770), (772, 1542), (1544, 2314)))
    assert pages == (1, 2, 3)


def test_a_chunk_that_merely_abuts_a_page_does_not_cite_it():
    """
    Half-open comparison. Using <= on either side pulls in the neighbour
    and every citation near a boundary comes out one page long.
    """
    pages, _ = _pages_for_span(0, 770, pdf_pages((0, 770), (770, 1540)))
    assert pages == (1,), "the chunk ends exactly where page 2 begins"


def test_a_document_without_a_page_map_has_no_pages():
    assert _pages_for_span(0, 100, None) == ((), 0)
    assert _pages_for_span(0, 100, []) == ((), 0)


def test_a_pdf_citation_names_a_page():
    citation = build_citation(hit(100, 400), FakeDocument(page_map=pdf_pages((0, 770))), "[1]")
    assert citation.locator == "page 1"
    assert citation.page_label == "page 1"


def test_a_pdf_citation_spanning_pages_names_the_range():
    document = FakeDocument(page_map=pdf_pages((0, 770), (772, 1542)))
    citation = build_citation(hit(615, 1200), document, "[1]")
    assert citation.locator == "pages 1-2"


def test_an_article_falls_back_to_a_character_range():
    """
    Still inside the document, not merely at it — an article has no pages
    to name, so the span is the finest locator available.
    """
    citation = build_citation(
        hit(1200, 2200), FakeDocument(page_map=None, source=DocumentSource.ARTICLE), "[1]"
    )
    assert citation.locator == "characters 1,200-2,200"
    assert citation.pages == ()


def test_a_scanned_image_is_not_described_as_page_1_of_1():
    document = FakeDocument(page_map=pdf_pages((0, 120)), source=DocumentSource.IMAGE)
    citation = build_citation(hit(0, 120), document, "[1]")
    assert citation.locator == "the scanned image"


def test_a_citation_into_an_ocred_page_says_so():
    """
    The difference between "the document says" and "a machine read this
    off a photo and believes it says". The user is entitled to know which.
    """
    document = FakeDocument(
        page_map=pdf_pages((0, 770), (772, 1542), ocr_pages=(1,), confidences={1: 54.0})
    )
    citation = build_citation(hit(800, 1200), document, "[1]")

    assert citation.from_ocr is True
    assert citation.ocr_confidence == 54.0


def test_a_citation_into_a_text_layer_is_not_marked_ocr():
    document = FakeDocument(page_map=pdf_pages((0, 770), (772, 1542), ocr_pages=(1,)))
    citation = build_citation(hit(100, 400), document, "[1]")
    assert citation.from_ocr is False
    assert citation.ocr_confidence is None


def test_confidence_across_pages_is_the_worst_one_not_the_average():
    """
    A citation is only as trustworthy as its least reliable page.
    Averaging would let one clean page disguise an unreadable one.
    """
    document = FakeDocument(
        page_map=pdf_pages(
            (0, 770),
            (772, 1542),
            ocr_pages=(0, 1),
            confidences={0: 95.0, 1: 41.0},
        )
    )
    citation = build_citation(hit(600, 1000), document, "[1]")

    assert citation.pages == (1, 2)
    assert citation.ocr_confidence == 41.0


def test_a_web_source_links_to_the_passage_not_the_page_top():
    document = FakeDocument(
        page_map=None,
        source=DocumentSource.ARTICLE,
        url="https://example.com/espresso",
    )
    citation = build_citation(
        hit(0, 80, "Milk texturing is a separate skill from extraction entirely"),
        document,
        "[1]",
    )

    assert citation.url.startswith("https://example.com/espresso#:~:text=")
    assert "Milk" in citation.url
    assert len(citation.url.split("text=")[1].split("%20")) <= FRAGMENT_WORDS


def test_a_source_with_no_url_has_no_link():
    citation = build_citation(hit(), FakeDocument(page_map=pdf_pages((0, 770))), "[1]")
    assert citation.url is None


def test_a_quote_too_short_to_identify_anything_links_plainly():
    """
    A fragment matching the first "the" on the page is worse than none.
    """
    document = FakeDocument(
        page_map=None, source=DocumentSource.ARTICLE, url="https://example.com/x"
    )
    citation = build_citation(hit(0, 5, "ok ok"), document, "[1]")
    assert citation.url == "https://example.com/x"


def test_a_long_chunk_is_quoted_at_a_word_boundary():
    content = " ".join(["invoice"] * 200)
    citation = build_citation(hit(0, len(content), content), FakeDocument(), "[1]")

    assert len(citation.quote) <= QUOTE_CHARS + 1  # the ellipsis
    assert citation.quote.endswith("…")
    assert not citation.quote.rstrip("…").endswith("invo"), "cut mid-word"


def test_newlines_are_flattened_so_a_prompt_block_stays_readable():
    citation = build_citation(
        hit(0, 40, "line one\n\nline  two\tline three"), FakeDocument(), "[1]"
    )
    assert citation.quote == "line one line two line three"


def test_an_untitled_document_falls_back_to_its_filename():
    document = FakeDocument(title=None, original_filename="march-invoice.pdf")
    assert build_citation(hit(), document, "[1]").title == "march-invoice.pdf"


def _citation(**overrides) -> Citation:
    base = dict(
        marker="[1]",
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        chunk_index=0,
        title="Invoice 2026-0042",
        source=DocumentSource.PDF,
        locator="page 3",
        pages=(3,),
        page_count=12,
        start_char=0,
        end_char=100,
        from_ocr=False,
        ocr_confidence=None,
        similarity=0.9,
        url=None,
        quote="Total 349.50 UAH",
    )
    return Citation(**{**base, **overrides})


def test_the_sources_block_names_marker_title_and_locator():
    rendered = render_sources([_citation()])

    assert "[1] Invoice 2026-0042 — page 3" in rendered
    assert "Total 349.50 UAH" in rendered


def test_the_sources_block_warns_the_model_about_ocr():
    """
    A model told a passage was read off a photo at 54% can hedge. One
    shown the same text bare cannot, and will state it as fact.
    """
    rendered = render_sources([_citation(from_ocr=True, ocr_confidence=54.0)])
    assert "read by OCR, 54% confidence" in rendered


def test_an_ocr_source_with_no_score_still_says_it_was_ocred():
    rendered = render_sources([_citation(from_ocr=True, ocr_confidence=None)])
    assert "(read by OCR)" in rendered


def test_no_sources_renders_something_a_prompt_can_use():
    """Not an empty string: a blank section reads as a formatting bug."""
    assert render_sources([]) == "(no sources)"


@pytest.mark.parametrize("count", [1, 3, 5])
def test_markers_are_sequential_with_no_gaps(count):
    """
    A gap would look to the model like a source it was given and should
    cite, which is how an answer ends up referencing [4] of three sources.
    """
    rendered = render_sources([_citation(marker=f"[{i + 1}]") for i in range(count)])
    for i in range(count):
        assert f"[{i + 1}]" in rendered
    assert f"[{count + 1}]" not in rendered
