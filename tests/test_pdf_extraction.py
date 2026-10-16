"""
Unit tests — the extractor is pure, so none of this needs a database or a
container. The PDFs are built here rather than committed as fixtures.
"""

import io

import pypdf
import pytest
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.services.ingestion.extractors.pdf import (
    MAX_PDF_PAGES,
    MIN_CHARS_PER_PAGE,
    PAGE_SEPARATOR,
    CorruptPdfError,
    EncryptedPdfError,
    PdfTooLargeError,
    extract_pdf_text,
    extract_pdf_text_async,
)

INVOICE = [
    "INVOICE 2026-0042",
    "Vendor: Kyiv Coffee Roasters",
    "Total: 480.00 UAH",
    "Date: 2026-03-11",
    "Payment terms: net 14 days from the invoice date",
]
PAGE_TWO = [
    "Page two of the invoice",
    "Line item: single origin beans 1kg",
    "Line item: delivery within Kyiv",
    "VAT included at twenty percent of the subtotal",
]


def text_pdf(*pages: list[str]) -> bytes:
    """A PDF whose pages carry real vector text."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(60, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def scan_pdf(*, stamp: str | None = None) -> bytes:
    """
    A scanned page: the words are pixels in an image, so there is no text
    layer at all. `stamp` adds real vector text on top, the way a page
    number or watermark does on a real scan.
    """
    image = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(image).text((80, 80), "RECEIPT 77/12 ATB Total 212.45 UAH", fill="black")
    png = io.BytesIO()
    image.save(png, "PNG")
    png.seek(0)

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.drawImage(ImageReader(png), 0, 0, width=A4[0], height=A4[1])
    if stamp:
        pdf.drawString(540, 20, stamp)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def joined(*pdfs: bytes) -> bytes:
    writer = pypdf.PdfWriter()
    for data in pdfs:
        writer.append(io.BytesIO(data))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def encrypted(password: str) -> bytes:
    writer = pypdf.PdfWriter()
    writer.append(io.BytesIO(text_pdf(INVOICE)))
    writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# the text layer


def test_text_is_extracted():
    result = extract_pdf_text(text_pdf(INVOICE))

    assert "INVOICE 2026-0042" in result.text
    assert "480.00 UAH" in result.text


def test_every_page_is_read():
    result = extract_pdf_text(text_pdf(INVOICE, PAGE_TWO))

    assert result.page_count == 2
    assert "single origin beans" in result.text


def test_pages_are_separated_in_the_combined_text():
    """The chunker splits on blank lines first, so boundaries must survive."""
    result = extract_pdf_text(text_pdf(INVOICE, PAGE_TWO))

    assert PAGE_SEPARATOR in result.text


def test_page_offsets_index_into_the_combined_text():
    """
    text[start_char:end_char] == page.text, exactly.

    This is what lets the OCR pass replace one page's slice instead of
    re-extracting the file — the same guarantee the chunker gives for
    chunks, for the same reason.
    """
    result = extract_pdf_text(text_pdf(INVOICE, PAGE_TWO, PAGE_TWO))

    assert result.page_count >= 2  # or the separator arithmetic is untested
    for page in result.pages:
        assert result.text[page.start_char : page.end_char] == page.text


def test_page_indexes_are_dense_and_ordered():
    result = extract_pdf_text(text_pdf(INVOICE, PAGE_TWO, INVOICE))

    assert [page.index for page in result.pages] == [0, 1, 2]


# the OCR verdict


def test_a_text_pdf_does_not_need_ocr():
    result = extract_pdf_text(text_pdf(INVOICE, PAGE_TWO))

    assert result.needs_ocr is False
    assert result.pages_needing_ocr == ()
    assert result.is_fully_scanned is False


def test_a_scan_is_flagged_rather_than_returning_silently_empty():
    """
    pypdf returns "" for an image-only page and raises nothing. Without the
    verdict this is an empty document that fails later for being "too
    short" — a reason unrelated to the cause.
    """
    result = extract_pdf_text(scan_pdf())

    assert result.text.strip() == ""
    assert result.needs_ocr is True
    assert result.pages_needing_ocr == (0,)
    assert result.is_fully_scanned is True


def test_a_scan_with_a_page_stamp_is_still_flagged():
    """
    The trap `if not text.strip()` falls into. A vector page number over
    the image makes the page look like it has a text layer, and the receipt
    ingests as the string "p. 1" — embedded, indexed, and empty of content.
    """
    result = extract_pdf_text(scan_pdf(stamp="p. 1"))

    assert result.text.strip() != ""  # there IS text, just not content
    assert result.needs_ocr is True


def test_a_mixed_pdf_flags_only_the_scanned_page():
    """A typed cover page in front of a scan is an ordinary file."""
    result = extract_pdf_text(joined(text_pdf(INVOICE), scan_pdf()))

    assert result.page_count == 2
    assert result.pages_needing_ocr == (1,)
    assert result.needs_ocr is True
    assert result.is_fully_scanned is False


def test_the_threshold_is_what_decides():
    """Pins the boundary, so changing MIN_CHARS_PER_PAGE is deliberate."""
    short = extract_pdf_text(text_pdf(["x" * (MIN_CHARS_PER_PAGE - 1)]))
    long = extract_pdf_text(text_pdf(["x" * (MIN_CHARS_PER_PAGE + 1)]))

    assert short.needs_ocr is True
    assert long.needs_ocr is False


def test_whitespace_does_not_count_towards_the_threshold():
    """A page of blank lines is not a text layer."""
    result = extract_pdf_text(text_pdf(["a b", "c d", "e f"]))

    assert result.pages[0].char_count < MIN_CHARS_PER_PAGE
    assert result.needs_ocr is True


# refusals


def test_a_corrupt_file_is_refused():
    with pytest.raises(CorruptPdfError):
        extract_pdf_text(b"%PDF-1.7\nthis is not a pdf at all\n")


def test_an_empty_file_is_refused():
    with pytest.raises(CorruptPdfError, match="empty"):
        extract_pdf_text(b"")


def test_a_non_pdf_is_refused():
    """An image posted to the PDF path must not reach the caller as a 500."""
    with pytest.raises(CorruptPdfError):
        extract_pdf_text(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)


def test_a_password_protected_pdf_is_refused():
    with pytest.raises(EncryptedPdfError, match="password"):
        extract_pdf_text(encrypted("secret"))


def test_an_owner_password_only_pdf_is_read():
    """
    Statements are routinely encrypted with only an owner password, which
    restricts printing rather than reading. Every viewer opens them with an
    empty user password, so refusing them would reject files the user does
    not consider protected.
    """
    result = extract_pdf_text(encrypted(""))

    assert "INVOICE 2026-0042" in result.text


def test_too_many_pages_is_refused():
    with pytest.raises(PdfTooLargeError, match="page limit"):
        extract_pdf_text(text_pdf(*([["page"]] * (MAX_PDF_PAGES + 1))))


def test_an_oversized_file_is_refused_before_parsing():
    """Checked on the byte length, so a hostile file is never opened."""
    with pytest.raises(PdfTooLargeError, match="byte limit"):
        extract_pdf_text(b"%PDF-1.7" + b"\x00" * (25 * 1024 * 1024))


# the async wrapper


@pytest.mark.anyio
async def test_the_async_wrapper_returns_the_same_result():
    data = text_pdf(INVOICE, PAGE_TWO)

    assert (await extract_pdf_text_async(data)).text == extract_pdf_text(data).text


@pytest.mark.anyio
async def test_the_async_wrapper_does_not_block_the_event_loop():
    """
    The reason the wrapper exists. BackgroundTasks share the server's loop,
    so awaiting pypdf directly would stall every other request for the
    duration. A counter running concurrently would stop advancing if the
    work were on the loop.
    """
    import anyio

    big = text_pdf(*([INVOICE] * 120))

    async def ticks_during(work) -> int:
        count = 0

        async def tick():
            nonlocal count
            while True:
                await anyio.sleep(0)
                count += 1

        async with anyio.create_task_group() as tg:
            tg.start_soon(tick)
            await work()
            tg.cancel_scope.cancel()
        return count

    async def on_the_loop():
        extract_pdf_text(big)  # synchronous, blocking this task

    async def in_a_thread():
        await extract_pdf_text_async(big)

    blocked = await ticks_during(on_the_loop)
    threaded = await ticks_during(in_a_thread)

    # Blocking starves the ticker almost entirely; the thread lets it run.
    assert threaded > blocked * 5, f"threaded={threaded} blocked={blocked}"