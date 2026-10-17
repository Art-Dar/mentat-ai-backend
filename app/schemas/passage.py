import uuid

from pydantic import BaseModel, ConfigDict

from app.models import DocumentSource
from app.services.retrieval.passage import Passage

STRICT = ConfigDict(extra="forbid")


class PassageResponse(BaseModel):
    """
    What a citation points at, with enough document around it to judge.

    `highlight_start` and `highlight_end` index `text`, not the document.
    Slice `text` with them and wrap the result — do not try to locate the
    quote by searching for it, because the same sentence can appear twice
    in one document and the second match is the wrong one.
    """

    model_config = STRICT

    document_id: uuid.UUID
    title: str | None
    source: DocumentSource

    text: str
    highlight_start: int
    highlight_end: int

    # Absolute offsets into the document, echoed back so a client can
    # re-request the same passage with a wider window without having to
    # remember what it asked for.
    start_char: int
    end_char: int

    # Whether there is more document in either direction. Both zero means
    # the window already covers the whole thing and "show more" should not
    # be offered.
    chars_before: int
    chars_after: int
    char_count: int

    # The same page numbers the citation carried, computed by the same
    # function, so the two cannot disagree.
    pages: list[int]
    page_count: int
    from_ocr: bool
    ocr_confidence: float | None

    @classmethod
    def from_passage(
        cls,
        passage: Passage,
        *,
        document_id: uuid.UUID,
        title: str | None,
        source: DocumentSource,
    ) -> "PassageResponse":
        return cls(
            document_id=document_id,
            title=title,
            source=source,
            text=passage.text,
            highlight_start=passage.highlight_start,
            highlight_end=passage.highlight_end,
            start_char=passage.start_char,
            end_char=passage.end_char,
            chars_before=passage.chars_before,
            chars_after=passage.chars_after,
            char_count=passage.char_count,
            pages=list(passage.provenance.pages),
            page_count=passage.provenance.page_count,
            from_ocr=passage.provenance.from_ocr,
            ocr_confidence=passage.provenance.ocr_confidence,
        )
