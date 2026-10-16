"""
Turns PDF bytes into text plus a per-page verdict on whether that page
needs OCR. Pure: no database, no HTTP, no settings. It is handed bytes and
returns a value (tested exhaustively without a container)
"""

import io
import logging
from dataclasses import dataclass

import anyio.to_thread
import pypdf
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# A 25 MB ceiling covers scanned multi-page statements with room to spare
# while keeping one upload from exhausting the worker's memory. Checked on
# the byte length BEFORE parsing, so a hostile file is never opened.
MAX_PDF_BYTES = 25 * 1024 * 1024

# pypdf parses lazily, so the cost is per page read, not per page present.
# The limit exists to bound the work, not the file.
MAX_PDF_PAGES = 500

# Non-whitespace characters below which a page is treated as unreadable.
# Calibrated against what stamps and watermarks actually produce: "p. 1"
# is 4, "1 of 3" is 6, a date stamp around 10, a letterhead line around 30.
# Raising this costs wasted OCR; lowering it loses scans silently.
MIN_CHARS_PER_PAGE = 32

# Keeps page boundaries visible to the chunker, which splits on blank lines
# before it splits mid-paragraph.
PAGE_SEPARATOR = "\n\n"


class PdfExtractionError(RuntimeError):
    """The file could not be read as a PDF. Never raised for a scan."""


class EncryptedPdfError(PdfExtractionError):
    """Protected with a user password, so the text cannot be reached."""


class CorruptPdfError(PdfExtractionError):
    """Not a readable PDF."""


class PdfTooLargeError(PdfExtractionError):
    """Over the byte or page budget."""


@dataclass(frozen=True)
class PageText:
    """
    One page's text, and where it sits in the combined text.

    start_char/end_char are what make the OCR ticket cheap: it can replace
    exactly this slice with the OCR result instead of re-extracting the
    whole file. Same guarantee the chunker gives for chunks, for the same
    reason.
    """

    index: int
    text: str
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        """Non-whitespace length. A page of blank lines is not a text layer."""
        return len(self.text.strip())

    @property
    def needs_ocr(self) -> bool:
        return self.char_count < MIN_CHARS_PER_PAGE


@dataclass(frozen=True)
class PdfExtraction:
    text: str
    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_needing_ocr(self) -> tuple[int, ...]:
        """Page indexes to hand to the OCR pass — not the whole document."""
        return tuple(page.index for page in self.pages if page.needs_ocr)

    @property
    def needs_ocr(self) -> bool:
        return bool(self.pages_needing_ocr)

    @property
    def is_fully_scanned(self) -> bool:
        """
        Every page is unreadable. Distinct from needs_ocr: a document that
        is partly readable is already useful and can be ingested now, with
        the scanned pages filled in later.
        """
        return bool(self.pages) and all(page.needs_ocr for page in self.pages)


def _open(data: bytes) -> pypdf.PdfReader:
    if len(data) > MAX_PDF_BYTES:
        raise PdfTooLargeError(f"PDF is {len(data)} bytes, over the {MAX_PDF_BYTES} byte limit")
    if not data:
        raise CorruptPdfError("PDF is empty")

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Statements and invoices are routinely encrypted with only an
            # OWNER password, which restricts printing rather than reading;
            # an empty user password opens them, and every viewer does this
            # silently. Refusing them outright would reject files the user
            # does not consider protected at all. decrypt() returns a
            # PasswordType enum whose NOT_DECRYPTED member is falsy.
            if not reader.decrypt(""):
                raise EncryptedPdfError("PDF is password-protected")
        page_count = len(reader.pages)
    except PdfReadError as exc:
        # The common base for PdfStreamError, EmptyFileError,
        # FileNotDecryptedError and WrongPasswordError.
        raise CorruptPdfError(f"{type(exc).__name__}: {exc}") from exc
    except EncryptedPdfError:
        raise
    except Exception as exc:
        # pypdf also raises plain ValueError/KeyError/struct.error on some
        # malformed files. A bad upload must not reach the caller as an
        # unhandled 500.
        raise CorruptPdfError(f"{type(exc).__name__}: {exc}") from exc

    if page_count > MAX_PDF_PAGES:
        raise PdfTooLargeError(f"PDF has {page_count} pages, over the {MAX_PDF_PAGES} page limit")
    if page_count == 0:
        raise CorruptPdfError("PDF has no pages")
    return reader


def extract_pdf_text(data: bytes) -> PdfExtraction:
    """
    Pull the text layer out of a PDF.
    """
    reader = _open(data)

    pages: list[PageText] = []
    parts: list[str] = []
    cursor = 0

    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # One unreadable page must not lose the other forty. An empty
            # string here is correct: the page then reads as needing OCR,
            # which is exactly what a page pypdf cannot parse does need.
            logger.warning("page %d could not be read, treating as a scan: %s", index, exc)
            text = ""

        if index:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)

        pages.append(
            PageText(index=index, text=text, start_char=cursor, end_char=cursor + len(text))
        )
        parts.append(text)
        cursor += len(text)

    return PdfExtraction(text="".join(parts), pages=tuple(pages))


async def extract_pdf_text_async(data: bytes) -> PdfExtraction:
    """
    called from the pipeline

    pypdf is synchronous C-bound work. Awaiting it directly on the event
    loop would stall every other request for the duration
    (Starlette's BackgroundTasks share the server's loop)
    """
    return await anyio.to_thread.run_sync(extract_pdf_text, data)
