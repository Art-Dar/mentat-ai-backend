import io
import logging
from dataclasses import dataclass

import anyio.to_thread
import pypdfium2 as pdfium

from app.services.ingestion.extractors.ocr import (
    DEFAULT_LANGUAGE,
    OcrEngineUnavailable,
    OcrError,
    extract_image_text,
)
from app.services.ingestion.extractors.pdf import (
    PAGE_SEPARATOR,
    PageText,
    PdfExtraction,
    extract_pdf_text,
)

logger = logging.getLogger(__name__)

OCR_RENDER_DPI = 200
MAX_OCR_PAGES = 20

# pypdfium2 renders at a scale factor relative to 72 dpi, the PDF unit.
_PDF_POINTS_PER_INCH = 72


@dataclass(frozen=True)
class PdfOcrResult:
    """
    What came back, and how much of it OCR had to supply
    """

    extraction: PdfExtraction
    ocr_pages: tuple[int, ...]
    skipped_pages: tuple[int, ...]
    mean_confidence: float | None

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)

    @property
    def still_needs_ocr(self) -> tuple[int, ...]:
        """
        Pages with no usable text even now: over the page budget, or OCR
        found nothing on them
        """
        return tuple(
            page.index
            for page in self.extraction.pages
            if page.needs_ocr and page.index not in self.ocr_pages
        )


def _render_page(document: pdfium.PdfDocument, index: int) -> bytes:
    """One page as PNG bytes, at the resolution OCR reads best."""
    page = document[index]
    image = page.render(scale=OCR_RENDER_DPI / _PDF_POINTS_PER_INCH).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _rebuild(pages: list[str]) -> PdfExtraction:
    """
    Reassemble text and offsets from scratch
    """
    rebuilt: list[PageText] = []
    parts: list[str] = []
    cursor = 0

    for index, text in enumerate(pages):
        if index:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        rebuilt.append(
            PageText(index=index, text=text, start_char=cursor, end_char=cursor + len(text))
        )
        parts.append(text)
        cursor += len(text)

    return PdfExtraction(text="".join(parts), pages=tuple(rebuilt))


def extract_pdf_text_with_ocr(data: bytes, *, language: str = DEFAULT_LANGUAGE) -> PdfOcrResult:
    """
    Pull the text layer out of a PDF, then OCR whatever pages lack one.
    """
    base = extract_pdf_text(data)

    if not base.needs_ocr:
        return PdfOcrResult(extraction=base, ocr_pages=(), skipped_pages=(), mean_confidence=None)

    wanted = base.pages_needing_ocr
    targets = wanted[:MAX_OCR_PAGES]
    skipped = wanted[MAX_OCR_PAGES:]
    if skipped:
        logger.warning(
            "PDF has %d pages needing OCR, over the %d page budget — pages %s left unread",
            len(wanted),
            MAX_OCR_PAGES,
            list(skipped),
        )

    texts = [page.text for page in base.pages]
    confidences: list[float] = []
    recovered: list[int] = []

    document = pdfium.PdfDocument(data)
    try:
        for index in targets:
            try:
                png = _render_page(document, index)
                result = extract_image_text(png, language=language)
            except OcrEngineUnavailable:
                # No engine, or no language pack. Every remaining page
                # would fail the same way, so stop rather than burn time
                # rediscovering it, and let the caller answer 503.
                raise
            except (OcrError, Exception) as exc:  # noqa: B014
                # One unreadable page must not lose the other nineteen.
                # The page keeps its empty text and stays in
                # still_needs_ocr, which is exactly what it is.
                logger.warning("OCR failed on page %d: %s", index, exc)
                continue

            if not result.has_text:
                # The page really is blank, or the scan is unreadable.
                # Not an error — a blank page in a scan is common.
                continue

            texts[index] = result.text
            recovered.append(index)
            if result.mean_confidence is not None:
                confidences.append(result.mean_confidence)
    finally:
        document.close()

    return PdfOcrResult(
        extraction=_rebuild(texts),
        ocr_pages=tuple(recovered),
        skipped_pages=tuple(skipped),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
    )


async def extract_pdf_text_with_ocr_async(
    data: bytes, *, language: str = DEFAULT_LANGUAGE
) -> PdfOcrResult:
    """
    Called from the API.
    """
    return await anyio.to_thread.run_sync(
        lambda: extract_pdf_text_with_ocr(data, language=language)
    )
