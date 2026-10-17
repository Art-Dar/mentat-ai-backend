import urllib.parse
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentSource
from app.services.retrieval.search import SearchHit

QUOTE_CHARS = 240
FRAGMENT_WORDS = 8


@dataclass(frozen=True)
class Citation:
    """
    One reference, ready to be rendered into a prompt or a UI.

    `marker` is what the model is asked to write inline, and what the
    client looks up to render a footnote. It is assigned by position in
    the result list, so it is stable only within one answer.
    """

    marker: str
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    title: str
    source: DocumentSource

    # Human-readable, already formatted: "page 12", "pages 3-4",
    # "characters 1,200-2,200". This is the string a UI shows.
    locator: str
    pages: tuple[int, ...]
    page_count: int

    start_char: int
    end_char: int

    from_ocr: bool
    ocr_confidence: float | None

    similarity: float
    # A link that opens at the passage where that is possible, not merely
    # at the document.
    url: str | None
    quote: str

    @property
    def page_label(self) -> str:
        if not self.pages:
            return ""
        if len(self.pages) == 1:
            return f"page {self.pages[0]}"
        return f"pages {self.pages[0]}-{self.pages[-1]}"


def _pages_for_span(
    start_char: int, end_char: int, page_map: list[dict] | None
) -> tuple[tuple[int, ...], int]:
    if not page_map:
        return (), 0

    touched = [
        int(page["index"]) + 1
        for page in page_map
        if start_char < int(page["end_char"]) and end_char > int(page["start_char"])
    ]
    return tuple(sorted(set(touched))), len(page_map)


def _ocr_provenance(
    pages: tuple[int, ...], page_map: list[dict] | None
) -> tuple[bool, float | None]:
    """
    Whether any cited page was machine-read, and how confidently.
    Confidence is the WORST of the cited pages, not the mean
    """
    if not page_map or not pages:
        return False, None

    cited = [page for page in page_map if int(page["index"]) + 1 in pages]
    from_ocr = any(page.get("source") == "ocr" for page in cited)
    scores = [
        float(page["mean_confidence"]) for page in cited if page.get("mean_confidence") is not None
    ]
    return from_ocr, (min(scores) if scores else None)


def _text_fragment_url(url: str | None, quote: str) -> str | None:
    """
    A link that scrolls to the passage, for sources that are web pages.
    """
    if not url:
        return None

    words = quote.split()[:FRAGMENT_WORDS]
    if len(words) < 3:
        # Too short to identify anything. A fragment that matches the
        # first "the" on the page is worse than none.
        return url

    return f"{url}#:~:text={urllib.parse.quote(' '.join(words))}"


def _locator(source: DocumentSource, pages: tuple[int, ...], start: int, end: int) -> str:
    if pages:
        if source == DocumentSource.IMAGE:
            # One image is one page; saying "page 1 of 1" is noise.
            return "the scanned image"
        return f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
    return f"characters {start:,}-{end:,}"


def _quote(content: str) -> str:
    flat = " ".join(content.split())
    if len(flat) <= QUOTE_CHARS:
        return flat
    # Cut at a word boundary so the excerpt does not end mid-word.
    return flat[:QUOTE_CHARS].rsplit(" ", 1)[0] + "…"


def build_citation(hit: SearchHit, document: Document, marker: str) -> Citation:
    """One citation. Separate from build_citations so it is testable alone."""
    pages, page_count = _pages_for_span(hit.start_char, hit.end_char, document.page_map)
    from_ocr, confidence = _ocr_provenance(pages, document.page_map)
    quote = _quote(hit.content)

    title = document.title or document.original_filename or "Untitled"

    return Citation(
        marker=marker,
        document_id=hit.document_id,
        chunk_id=hit.chunk_id,
        chunk_index=hit.chunk_index,
        title=title,
        source=document.source,
        locator=_locator(document.source, pages, hit.start_char, hit.end_char),
        pages=pages,
        page_count=page_count,
        start_char=hit.start_char,
        end_char=hit.end_char,
        from_ocr=from_ocr,
        ocr_confidence=confidence,
        similarity=hit.similarity,
        url=_text_fragment_url(document.url, quote),
        quote=quote,
    )


async def build_citations(db: AsyncSession, hits: list[SearchHit]) -> list[Citation]:
    """
    Citations for a page of search hits, in the order they were ranked.
    """
    if not hits:
        return []

    document_ids = {hit.document_id for hit in hits}
    rows = (await db.execute(select(Document).where(Document.id.in_(document_ids)))).scalars().all()
    documents = {document.id: document for document in rows}

    citations: list[Citation] = []
    for hit in hits:
        document = documents.get(hit.document_id)
        if document is None:
            continue
        # Numbered by position among the citations that survived, so the
        # markers are always 1..n with no gaps — a gap would look to the
        # model like a source it was not given.
        citations.append(build_citation(hit, document, f"[{len(citations) + 1}]"))

    return citations


def render_sources(citations: list[Citation]) -> str:
    """
    The sources block for citation to drop into its prompt.
    """
    if not citations:
        return "(no sources)"

    blocks: list[str] = []
    for citation in citations:
        header = f"{citation.marker} {citation.title} — {citation.locator}"
        if citation.from_ocr:
            if citation.ocr_confidence is not None:
                header += f" (read by OCR, {citation.ocr_confidence:.0f}% confidence)"
            else:
                header += " (read by OCR)"
        blocks.append(f"{header}\n{citation.quote}")

    return "\n\n".join(blocks)
