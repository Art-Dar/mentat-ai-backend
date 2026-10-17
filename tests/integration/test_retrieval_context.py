"""
The retrieval attribute: grouping citations into what a UI draws.

Needs a database because it reads documents and their tags. The grouping
logic is what matters: five citations from one invoice must become ONE
chip, carrying every page they touched and every marker that points into
it, so the client can highlight the chip and the footnote together.
"""

import uuid

import pytest
from sqlalchemy import delete, event, select

from app.core.db import AsyncSessionLocal, engine
from app.models import (
    Document,
    DocumentSource,
    DocumentTag,
    IngestionStatus,
    Tag,
    User,
)
from app.models.user import AuthIdentity, AuthProvider
from app.schemas.answer import RetrievalRead
from app.services.answering.answering_context import build_retrieval_context
from app.services.retrieval.citations import build_citations
from app.services.retrieval.search import SearchHit

PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 770, "needs_ocr": False, "source": "pdf"},
    {"index": 1, "start_char": 772, "end_char": 1542, "needs_ocr": False,
     "source": "ocr", "mean_confidence": 61.0},
]
CONTENT = "x" * 1542


@pytest.fixture
async def owner(keep_rows):
    email = f"ctx-{uuid.uuid4().hex[:8]}@example.com"
    async with AsyncSessionLocal() as db:
        user = User(email=email, email_verified=True)
        user.identities.append(
            AuthIdentity(
                provider=AuthProvider.PASSWORD,
                provider_user_id=email,
                password_hash="not-used-here",
            )
        )
        db.add(user)
        await db.commit()
        user_id = user.id

    yield user_id

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(User).where(User.id == user_id))
        await db.commit()


@pytest.fixture
async def library(owner, keep_rows):
    """
    Two documents: a two-page PDF tagged Finance + Coffee, and an
    untagged scanned image.
    """
    created_tags: list[uuid.UUID] = []
    async with AsyncSessionLocal() as db:
        invoice = Document(
            user_id=owner,
            source=DocumentSource.PDF,
            status=IngestionStatus.COMPLETED,
            content=CONTENT,
            title="Invoice 2026-0042",
            storage_key=f"documents/{owner}/{uuid.uuid4()}.pdf",
            file_size=1,
            page_map=PAGE_MAP,
            url="https://example.com/invoice",
        )
        receipt = Document(
            user_id=owner,
            source=DocumentSource.IMAGE,
            status=IngestionStatus.COMPLETED,
            content="y" * 200,
            title="ATB receipt",
            storage_key=f"documents/{owner}/{uuid.uuid4()}.png",
            file_size=1,
            page_map=[
                {"index": 0, "start_char": 0, "end_char": 200, "needs_ocr": False,
                 "source": "ocr", "mean_confidence": 54.0}
            ],
        )
        db.add_all([invoice, receipt])
        await db.flush()

        for name in ("Finance", "Coffee"):
            tag = Tag(id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}")
            db.add(tag)
            await db.flush()
            created_tags.append(tag.id)
            db.add(DocumentTag(document_id=invoice.id, tag_id=tag.id))

        await db.commit()
        ids = (invoice.id, receipt.id)

    yield ids

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id.in_(ids)))
        await db.execute(delete(Tag).where(Tag.id.in_(created_tags)))
        await db.commit()


def hit(document_id, start, end, index=0):
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=index,
        content="Total 349.50 UAH payable within fourteen days of issue",
        start_char=start,
        end_char=end,
        similarity=0.8,
        document_title=None,
        document_url=None,
    )


@pytest.mark.anyio
async def test_several_citations_from_one_document_become_one_entry(library):
    """
    Five footnotes from one invoice, one chip. The alternative is five
    identical chips, which is what a naive client would render.
    """
    invoice, _ = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(
            db, [hit(invoice, 100, 400), hit(invoice, 800, 1200, index=1)]
        )
        context = await build_retrieval_context(db, citations)

    assert context.chunk_count == 2
    assert context.document_count == 1

    entry = context.documents[0]
    assert entry.markers == ("[1]", "[2]")
    # The union of both citations' pages, so one chip covers everything
    # this document contributed.
    assert entry.pages == (1, 2)
    assert entry.locator == "pages 1-2"


@pytest.mark.anyio
async def test_a_document_is_flagged_ocr_if_any_cited_passage_was(library):
    """
    Page 1 is a text layer, page 2 was OCRed. The chip must carry the
    warning: part of what the answer used is uncertain.
    """
    invoice, _ = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(invoice, 100, 400)])
        clean = await build_retrieval_context(db, citations)

        citations = await build_citations(db, [hit(invoice, 100, 400), hit(invoice, 800, 1200)])
        mixed = await build_retrieval_context(db, citations)

    assert clean.documents[0].from_ocr is False
    assert mixed.documents[0].from_ocr is True


@pytest.mark.anyio
async def test_tags_say_where_in_the_knowledge_base_the_answer_came_from(library):
    invoice, _ = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(invoice, 100, 400)])
        context = await build_retrieval_context(db, citations)

    names = [tag.name for tag in context.tags]
    assert len(names) == 2
    assert names == sorted(names, key=str.lower), "tags should arrive sorted"
    assert all(tag.color.startswith("#") for tag in context.tags)
    assert context.documents[0].tags == context.tags


@pytest.mark.anyio
async def test_an_untagged_document_simply_has_no_tags(library):
    _, receipt = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(receipt, 0, 150)])
        context = await build_retrieval_context(db, citations)

    assert context.documents[0].tags == ()
    assert context.tags == ()


@pytest.mark.anyio
async def test_documents_keep_relevance_order_not_alphabetical(library):
    """
    The first chip should be the best match. Sorting by title would put
    it wherever the alphabet happens to place it.
    """
    invoice, receipt = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(receipt, 0, 150), hit(invoice, 100, 400)])
        context = await build_retrieval_context(db, citations)

    assert [d.document_id for d in context.documents] == [receipt, invoice]


@pytest.mark.anyio
async def test_has_file_tells_the_client_whether_a_download_exists(library):
    invoice, _ = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(invoice, 100, 400)])
        context = await build_retrieval_context(db, citations)

    assert context.documents[0].has_file is True
    assert context.documents[0].url == "https://example.com/invoice"


@pytest.mark.anyio
async def test_the_query_count_does_not_grow_with_the_number_of_documents(library):
    """
    The absolute count is an implementation detail — selectinload issues
    its own second query for the related tags, so it is three, not two.
    """
    invoice, receipt = library

    def count_selects(record_into: list[str]):
        def record(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                record_into.append(statement)
        return record

    async def queries_for(hits) -> int:
        statements: list[str] = []
        async with AsyncSessionLocal() as db:
            citations = await build_citations(db, hits)
            listener = count_selects(statements)
            event.listen(engine.sync_engine, "before_cursor_execute", listener)
            try:
                await build_retrieval_context(db, citations)
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", listener)
        return len(statements)

    one = await queries_for([hit(invoice, 100, 400)])
    many = await queries_for(
        [hit(invoice, 100, 400), hit(invoice, 800, 1200), hit(receipt, 0, 150)]
    )

    assert many == one, (
        f"{many} queries for 3 citations across 2 documents vs {one} for 1 — "
        "the context is querying per document"
    )
    assert one <= 3, "more than three queries means something else regressed too"


@pytest.mark.anyio
async def test_no_citations_is_an_empty_context_not_a_crash():
    async with AsyncSessionLocal() as db:
        context = await build_retrieval_context(db, [])

    assert context.citations == ()
    assert context.documents == ()
    assert context.tags == ()
    assert context.chunk_count == 0


@pytest.mark.anyio
async def test_the_schema_carries_everything_the_frontend_needs(library):
    """
    Serialised here so a change to the dataclasses that breaks
    the wire format fails now rather than in the extension.
    """
    invoice, receipt = library

    async with AsyncSessionLocal() as db:
        citations = await build_citations(
            db, [hit(invoice, 800, 1200), hit(receipt, 0, 150)]
        )
        context = await build_retrieval_context(db, citations)

    payload = RetrievalRead.from_context(context).model_dump(mode="json")

    assert payload["chunk_count"] == 2
    assert payload["document_count"] == 2
    assert {c["marker"] for c in payload["citations"]} == {"[1]", "[2]"}

    first = payload["documents"][0]
    assert first["markers"] == ["[1]"]
    assert first["locator"] == "page 2"
    assert first["from_ocr"] is True
    assert first["has_file"] is True
    assert [t["name"] for t in first["tags"]] == [t["name"] for t in payload["tags"]]

    cited = payload["citations"][0]
    assert cited["ocr_confidence"] == 61.0
    assert cited["quote"]
    assert cited["pages"] == [2]


@pytest.mark.anyio
async def test_a_chunk_row_survives_the_whole_chain(library):
    """
    End to end from stored rows: search hit to citation to context to
    wire format, so the column names, the page_map keys and the schema
    field names are checked against each other rather than assumed.
    """
    invoice, _ = library

    async with AsyncSessionLocal() as db:
        document = (
            await db.execute(select(Document).where(Document.id == invoice))
        ).scalars().one()
        citations = await build_citations(
            db, [hit(document.id, 0, 300)]
        )
        payload = RetrievalRead.from_context(
            await build_retrieval_context(db, citations)
        ).model_dump(mode="json")

    assert payload["citations"][0]["title"] == "Invoice 2026-0042"
    assert payload["citations"][0]["source"] == "pdf"
    assert payload["documents"][0]["page_count"] == 2