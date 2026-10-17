import uuid

import pytest
from sqlalchemy import delete, event, select

from app.core.db import AsyncSessionLocal, engine
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.models.user import AuthIdentity, AuthProvider
from app.services.retrieval.citations import build_citations
from app.services.retrieval.search import SearchHit

PAGE_MAP = [
    {"index": 0, "start_char": 0, "end_char": 770, "needs_ocr": False, "source": "pdf"},
    {
        "index": 1,
        "start_char": 772,
        "end_char": 1542,
        "needs_ocr": False,
        "source": "ocr",
        "mean_confidence": 61.0,
    },
]
CONTENT = "x" * 1542


@pytest.fixture
async def owner(keep_rows):
    email = f"cite-{uuid.uuid4().hex[:8]}@example.com"
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
async def invoice(owner, keep_rows):
    async with AsyncSessionLocal() as db:
        document = Document(
            user_id=owner,
            source=DocumentSource.PDF,
            status=IngestionStatus.COMPLETED,
            content=CONTENT,
            title="Invoice 2026-0042",
            original_filename="invoice.pdf",
            storage_key=f"documents/{owner}/{uuid.uuid4()}.pdf",
            file_size=1,
            page_map=PAGE_MAP,
        )
        db.add(document)
        await db.commit()
        document_id = document.id

    yield document_id

    if keep_rows:
        return
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Document).where(Document.id == document_id))
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
        document_title="Invoice 2026-0042",
        document_url=None,
    )


@pytest.mark.anyio
async def test_citations_carry_the_page_and_the_provenance(invoice):
    async with AsyncSessionLocal() as db:
        citations = await build_citations(
            db, [hit(invoice, 100, 400), hit(invoice, 800, 1200, index=1)]
        )

    assert [c.marker for c in citations] == ["[1]", "[2]"]

    first, second = citations
    assert first.locator == "page 1"
    assert first.from_ocr is False

    assert second.locator == "page 2"
    assert second.from_ocr is True, "page 2 was OCRed and the citation must say so"
    assert second.ocr_confidence == 61.0

    assert all(c.title == "Invoice 2026-0042" for c in citations)
    assert all(c.page_count == 2 for c in citations)


@pytest.mark.anyio
async def test_several_hits_from_one_document_cost_one_query(invoice):
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        async with AsyncSessionLocal() as db:
            citations = await build_citations(
                db, [hit(invoice, i * 100, i * 100 + 200) for i in range(5)]
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)

    assert len(citations) == 5
    assert len(statements) == 1, f"expected one SELECT, got {len(statements)}"


@pytest.mark.anyio
async def test_a_hit_whose_document_vanished_is_dropped_not_fatal(invoice):
    """
    A document can be deleted between the search and the citation. Losing
    one reference beats failing an answer that is otherwise correct.
    """
    async with AsyncSessionLocal() as db:
        citations = await build_citations(db, [hit(invoice, 100, 400), hit(uuid.uuid4(), 0, 100)])

    assert len(citations) == 1
    assert citations[0].marker == "[1]", "markers must stay gapless after a drop"


@pytest.mark.anyio
async def test_no_hits_needs_no_query():
    async with AsyncSessionLocal() as db:
        assert await build_citations(db, []) == []


@pytest.mark.anyio
async def test_offsets_in_a_citation_still_index_the_document(invoice):
    """
    The guarantee the whole feature rests on: a reader following the
    citation must land on the text the answer was drawn from.
    """
    async with AsyncSessionLocal() as db:
        citation = (await build_citations(db, [hit(invoice, 100, 400)]))[0]
        document = await db.get(Document, invoice)

    assert len(document.content[citation.start_char : citation.end_char]) == 300


@pytest.mark.anyio
async def test_a_real_chunk_row_maps_onto_a_page(invoice, owner):
    """
    End to end from a stored Chunk rather than a hand-built SearchHit, so
    the column names in chunks and in page_map are checked against each
    other rather than assumed.
    """
    async with AsyncSessionLocal() as db:
        chunk = Chunk(
            document_id=invoice,
            chunk_index=0,
            content=CONTENT[900:1100],
            start_char=900,
            end_char=1100,
            embedding=[0.01] * EMBEDDING_DIM,
            embedding_model="fake",
        )
        db.add(chunk)
        await db.commit()
        stored = (
            (await db.execute(select(Chunk).where(Chunk.document_id == invoice))).scalars().one()
        )

        citations = await build_citations(
            db,
            [
                SearchHit(
                    chunk_id=stored.id,
                    document_id=invoice,
                    chunk_index=stored.chunk_index,
                    content=stored.content,
                    start_char=stored.start_char,
                    end_char=stored.end_char,
                    similarity=0.9,
                    document_title=None,
                    document_url=None,
                )
            ],
        )

    assert citations[0].pages == (2,)
    assert citations[0].from_ocr is True
