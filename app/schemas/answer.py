import uuid

from pydantic import BaseModel, ConfigDict

from app.models import DocumentSource
from app.services.answering.answering_context import RetrievalContext

STRICT = ConfigDict(extra="forbid")


class TagRef(BaseModel):
    model_config = STRICT

    id: uuid.UUID
    name: str
    color: str


class CitationRead(BaseModel):
    model_config = STRICT

    marker: str
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    title: str
    source: DocumentSource
    locator: str
    pages: list[int]
    page_count: int
    start_char: int
    end_char: int
    from_ocr: bool
    ocr_confidence: float | None
    similarity: float
    # Opens at the passage where that is possible, not just the document.
    url: str | None
    quote: str


class DocumentRef(BaseModel):
    model_config = STRICT

    document_id: uuid.UUID
    title: str
    source: DocumentSource
    markers: list[str]
    pages: list[int]
    page_count: int
    locator: str
    has_file: bool
    url: str | None
    from_ocr: bool
    tags: list[TagRef]


class RetrievalRead(BaseModel):
    model_config = STRICT

    citations: list[CitationRead]
    documents: list[DocumentRef]
    tags: list[TagRef]
    document_count: int
    chunk_count: int

    @classmethod
    def from_context(cls, context: RetrievalContext) -> "RetrievalRead":
        return cls(
            citations=[
                CitationRead(
                    marker=c.marker,
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    title=c.title,
                    source=c.source,
                    locator=c.locator,
                    pages=list(c.pages),
                    page_count=c.page_count,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    from_ocr=c.from_ocr,
                    ocr_confidence=c.ocr_confidence,
                    similarity=c.similarity,
                    url=c.url,
                    quote=c.quote,
                )
                for c in context.citations
            ],
            documents=[
                DocumentRef(
                    document_id=d.document_id,
                    title=d.title,
                    source=d.source,
                    markers=list(d.markers),
                    pages=list(d.pages),
                    page_count=d.page_count,
                    locator=d.locator,
                    has_file=d.has_file,
                    url=d.url,
                    from_ocr=d.from_ocr,
                    tags=[TagRef(id=t.id, name=t.name, color=t.color) for t in d.tags],
                )
                for d in context.documents
            ],
            tags=[TagRef(id=t.id, name=t.name, color=t.color) for t in context.tags],
            document_count=context.document_count,
            chunk_count=context.chunk_count,
        )