import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Document, DocumentSource, DocumentTag
from app.services.retrieval.citations import Citation


@dataclass(frozen=True)
class TagReference:
    id: uuid.UUID
    name: str
    color: str


@dataclass(frozen=True)
class DocumentReference:
    """
    One source document, however many chunks of it were cited.

    `markers` is what ties this back to the prose: a client can highlight
    this chip when the reader hovers [2], and highlight [2] when they
    hover the chip.
    """

    document_id: uuid.UUID
    title: str
    source: DocumentSource
    markers: tuple[str, ...]
    pages: tuple[int, ...]
    page_count: int
    # True when the original bytes are in object storage, so the client
    # knows whether GET /documents/{id}/file will give it something.
    has_file: bool
    url: str | None
    from_ocr: bool
    tags: tuple[TagReference, ...]

    @property
    def locator(self) -> str:
        """How to describe this document's contribution in one phrase."""
        if not self.pages:
            return ""
        if self.source == DocumentSource.IMAGE:
            return "scanned image"
        if len(self.pages) == 1:
            return f"page {self.pages[0]}"
        return f"pages {self.pages[0]}-{self.pages[-1]}"


@dataclass(frozen=True)
class RetrievalContext:
    """What the answer was built from. Never what the answer says."""

    citations: tuple[Citation, ...]
    documents: tuple[DocumentReference, ...]
    tags: tuple[TagReference, ...]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.citations)


async def _tags_for_documents(
    db: AsyncSession, document_ids: set[uuid.UUID]
) -> dict[uuid.UUID, list[TagReference]]:
    """
    Tags for several documents in one query.
    """
    if not document_ids:
        return {}

    rows = (
        (
            await db.execute(
                select(DocumentTag)
                .where(DocumentTag.document_id.in_(document_ids))
                .options(selectinload(DocumentTag.tag))
            )
        )
        .scalars()
        .all()
    )

    grouped: dict[uuid.UUID, list[TagReference]] = {}
    for link in rows:
        grouped.setdefault(link.document_id, []).append(
            TagReference(id=link.tag.id, name=link.tag.name, color=link.tag.color)
        )
    for tags in grouped.values():
        tags.sort(key=lambda tag: tag.name.lower())
    return grouped


async def build_retrieval_context(
    db: AsyncSession, citations: tuple[Citation, ...] | list[Citation]
) -> RetrievalContext:
    """
    Group citations into the shape a client renders.
    """
    citations = tuple(citations)
    if not citations:
        return RetrievalContext(citations=(), documents=(), tags=())

    document_ids = {citation.document_id for citation in citations}

    rows = (await db.execute(select(Document).where(Document.id.in_(document_ids)))).scalars().all()
    documents = {document.id: document for document in rows}
    tags_by_document = await _tags_for_documents(db, document_ids)

    references: list[DocumentReference] = []
    seen: set[uuid.UUID] = set()

    for citation in citations:
        if citation.document_id in seen:
            continue
        seen.add(citation.document_id)

        mine = [c for c in citations if c.document_id == citation.document_id]
        document = documents.get(citation.document_id)

        references.append(
            DocumentReference(
                document_id=citation.document_id,
                title=citation.title,
                source=citation.source,
                markers=tuple(c.marker for c in mine),
                # The union of every cited page, so one chip covers all
                # of this document's contributions.
                pages=tuple(sorted({page for c in mine for page in c.pages})),
                page_count=citation.page_count,
                has_file=bool(document is not None and document.storage_key),
                url=document.url if document is not None else None,
                # True when ANY cited passage from this document was
                # machine-read: the chip should carry the warning if any
                # part of what was used is uncertain.
                from_ocr=any(c.from_ocr for c in mine),
                tags=tuple(tags_by_document.get(citation.document_id, ())),
            )
        )

    unique_tags: dict[uuid.UUID, TagReference] = {}
    for reference in references:
        for tag in reference.tags:
            unique_tags.setdefault(tag.id, tag)

    return RetrievalContext(
        citations=citations,
        documents=tuple(references),
        tags=tuple(sorted(unique_tags.values(), key=lambda tag: tag.name.lower())),
    )
