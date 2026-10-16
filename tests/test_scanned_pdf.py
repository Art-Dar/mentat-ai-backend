import io
import os
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.services.ingestion.extractors.scanned_pdf import (
    MAX_OCR_PAGES,
    OCR_RENDER_DPI,
    extract_pdf_text_with_ocr,
    extract_pdf_text_with_ocr_async,
)

_HAVE_TESSERACT = shutil.which("tesseract") is not None

needs_tesseract = pytest.mark.skipif(
    not _HAVE_TESSERACT and os.getenv("REQUIRE_TESSERACT") != "1",
    reason=(
        "the tesseract binary is not installed on this host "
        "(set REQUIRE_TESSERACT=1 to fail instead of skip)"
    ),
)

RECEIPT = ["ATB Market", "Chek 10482", "Kava zernova 1kg 249.00", "Razom: 349.50 UAH"]
TYPED = [
    "INVOICE 2026-0042",
    "Vendor: Kyiv Coffee Roasters",
    "Total: 480.00 UAH",
    "Payment terms: net 14 days from the invoice date",
]


def _font(size: int = 30):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    pytest.skip("no scalable font available to render a test scan")


def _page_image(lines: list[str]) -> io.BytesIO:
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    y = 120
    for line in lines:
        draw.text((100, y), line, fill="black", font=font)
        y += 60
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)
    return buffer


def build_pdf(*pages: tuple[str, list[str]]) -> bytes:
    """
    Each page is ("text", lines) for a real text layer or ("scan", lines)
    for a full-page image, which is what a scanner app produces.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for kind, lines in pages:
        if kind == "scan":
            pdf.drawImage(ImageReader(_page_image(lines)), 0, 0, width=A4[0], height=A4[1])
        else:
            y = 800
            for line in lines:
                pdf.drawString(60, y, line)
                y -= 20
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def assert_offsets_hold(extraction):
    """
    Every page must still index its own slice of the combined text
    """
    for page in extraction.pages:
        assert extraction.text[page.start_char : page.end_char] == page.text, (
            f"page {page.index} offsets do not match its text"
        )


def test_a_text_pdf_is_returned_unchanged_without_starting_the_engine(monkeypatch):
    """
    The common case must not pay for the rare one. If this ever calls the
    extractor, every PDF upload gets slower for no benefit.
    """

    def explode(*args, **kwargs):
        raise AssertionError("OCR was started for a PDF that has a text layer")

    monkeypatch.setattr("app.services.ingestion.extractors.scanned_pdf.extract_image_text", explode)

    result = extract_pdf_text_with_ocr(build_pdf(("text", TYPED)))

    assert result.ocr_pages == ()
    assert result.used_ocr is False
    assert result.mean_confidence is None
    assert "480.00" in result.extraction.text
    assert_offsets_hold(result.extraction)


def test_render_dpi_is_the_measured_optimum():
    """
    Rendering the same receipt at 150/200/250/300/400 dpi
    gave identical character counts with mean confidence
    87/92/87/88/85 and cost rising from 1.6s to 3.6s
    """
    assert OCR_RENDER_DPI == 200


@needs_tesseract
def test_a_fully_scanned_receipt_is_now_readable():
    """The exact case that failed silently before this module existed."""
    result = extract_pdf_text_with_ocr(build_pdf(("scan", RECEIPT)))

    assert result.used_ocr
    assert result.ocr_pages == (0,)
    assert "349.50" in result.extraction.text
    assert result.mean_confidence is not None and result.mean_confidence > 60
    assert result.still_needs_ocr == (), "the page was recovered, so nothing is outstanding"
    assert_offsets_hold(result.extraction)


@needs_tesseract
def test_a_mixed_pdf_ocrs_only_the_scanned_pages():
    data = build_pdf(("text", TYPED), ("scan", ["DODATOK A", "Dostavka 60.00 UAH"]))
    result = extract_pdf_text_with_ocr(data)

    assert result.ocr_pages == (1,), "only the picture page should have been recognised"
    assert "480.00" in result.extraction.text, "the typed page must survive"
    assert "60.00" in result.extraction.text, "the scanned page must be added"
    assert_offsets_hold(result.extraction)


@needs_tesseract
def test_offsets_are_rebuilt_not_patched():
    """
    The subtle one. Page 0 is scanned, so OCR replaces empty text with
    real text and everything after it shifts. If the offsets were patched
    instead of recomputed, page 1 would point at the wrong characters and
    every citation into this document would be wrong.
    """
    data = build_pdf(("scan", RECEIPT), ("text", TYPED))
    result = extract_pdf_text_with_ocr(data)

    assert result.ocr_pages == (0,)
    assert_offsets_hold(result.extraction)

    page_one = result.extraction.pages[1]
    assert page_one.start_char > 0, "page 1 must have moved along after page 0 grew"
    assert "480.00" in result.extraction.text[page_one.start_char : page_one.end_char]


@needs_tesseract
def test_pages_over_the_budget_are_reported_not_dropped(monkeypatch):
    """
    A 200-page scanned archive must not hold a worker for ten minutes.
    The pages past the budget stay visible in skipped_pages, so the API
    can say so and a later ticket can queue them.
    """
    monkeypatch.setattr("app.services.ingestion.extractors.scanned_pdf.MAX_OCR_PAGES", 1)

    data = build_pdf(("scan", RECEIPT), ("scan", ["DRUHA STORINKA", "Suma 120.00"]))
    result = extract_pdf_text_with_ocr(data)

    assert result.ocr_pages == (0,)
    assert result.skipped_pages == (1,)
    assert 1 in result.still_needs_ocr
    assert_offsets_hold(result.extraction)


@needs_tesseract
def test_one_unreadable_page_does_not_lose_the_others(monkeypatch):
    """A single bad page is not a reason to fail a whole document."""
    calls = {"n": 0}
    real = extract_pdf_text_with_ocr.__globals__["extract_image_text"]

    def flaky(data, *, language="eng"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pretend this page is corrupt")
        return real(data, language=language)

    monkeypatch.setattr("app.services.ingestion.extractors.scanned_pdf.extract_image_text", flaky)

    data = build_pdf(("scan", RECEIPT), ("scan", ["DRUHA STORINKA", "Suma 120.00 UAH"]))
    result = extract_pdf_text_with_ocr(data)

    assert result.ocr_pages == (1,), "the second page should still have been read"
    assert 0 in result.still_needs_ocr
    assert "120.00" in result.extraction.text
    assert_offsets_hold(result.extraction)


@needs_tesseract
async def test_the_async_wrapper_agrees_with_the_sync_one(anyio_backend):
    data = build_pdf(("scan", RECEIPT))
    assert (
        await extract_pdf_text_with_ocr_async(data)
    ).extraction.text == extract_pdf_text_with_ocr(data).extraction.text


def test_the_page_budget_is_a_real_limit():
    assert 0 < MAX_OCR_PAGES <= 50, "a budget nobody can hit is not a budget"
