import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import get_embedder
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.llm.fake_client import FakeChatClient
from app.services.llm.groq_client import get_chat_client

INVOICE = (
    "Invoice 2026-0042 from Kyiv Coffee Roasters. Payment terms: net 14 days "
    "from the invoice date. Total: 480.00 UAH including VAT at 20 percent."
)


@pytest.fixture
def fake_embedder():
    fake = FakeEmbedder()
    app.dependency_overrides[get_embedder] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_embedder, None)


@pytest.fixture
def chat():
    holder: dict = {"client": FakeChatClient("Net 14 days [1].")}
    app.dependency_overrides[get_chat_client] = lambda: holder["client"]

    def use(client: FakeChatClient) -> FakeChatClient:
        holder["client"] = client
        return client

    yield use
    app.dependency_overrides.pop(get_chat_client, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def stranger_headers(client, db_session):
    """A second signed-in account, for the isolation tests."""
    email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    password = "correct-horse-battery-staple"
    response = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert response.status_code in (200, 201), response.text
    token = client.post("/api/v1/auth/login", json={"email": email, "password": password}).json()[
        "access_token"
    ]

    yield {"Authorization": f"Bearer {token}"}

    user = db_session.execute(select(User).where(User.email == email)).scalars().one_or_none()
    if user is not None:
        db_session.delete(user)
        db_session.commit()


@pytest.fixture
def document(db_session, seeded_user):
    email, _ = seeded_user
    user = db_session.execute(select(User).where(User.email == email)).scalars().one()
    doc = Document(
        id=uuid.uuid4(),
        user_id=user.id,
        source=DocumentSource.PDF,
        status=IngestionStatus.COMPLETED,
        content=INVOICE,
        title="Invoice 2026-0042",
        page_map=[
            {
                "index": 0,
                "start_char": 0,
                "end_char": len(INVOICE),
                "needs_ocr": False,
                "source": "pdf",
            }
        ],
    )
    db_session.add(doc)
    db_session.flush()
    db_session.add(
        Chunk(
            document_id=doc.id,
            chunk_index=0,
            content=INVOICE,
            start_char=0,
            end_char=len(INVOICE),
            embedding=[0.01] * EMBEDDING_DIM,
            embedding_model="fake",
        )
    )
    db_session.commit()

    yield doc

    existing = db_session.get(Document, doc.id)
    if existing is not None:
        db_session.delete(existing)
        db_session.commit()


def start(client, headers, **body):
    return client.post("/api/v1/conversations", headers=headers, json=body)


def ask(client, headers, conversation_id=None, **body):
    body.setdefault("question", "when is the invoice due?")
    body.setdefault("min_similarity", -1.0)
    if conversation_id is not None:
        body["conversation_id"] = str(conversation_id)
    return client.post("/api/v1/ask", headers=headers, json=body)


def test_a_thread_is_created_empty_and_titled_by_its_first_question(
    client, auth_headers, document, fake_embedder, chat
):
    """
    No model call for a sidebar label: the first question is a better
    title than anything a user types up front, and free.
    """
    created = start(client, auth_headers)
    assert created.status_code == 201
    assert created.json()["title"] is None
    conversation_id = created.json()["id"]

    ask(client, auth_headers, conversation_id, question="when is the invoice due?")

    detail = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).json()
    assert detail["title"] == "when is the invoice due?"
    assert detail["message_count"] == 2
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_a_one_shot_question_stores_nothing(client, auth_headers, document, fake_embedder, chat):
    """
    /ask without a conversation is the extension's quick lookup. It must
    not silently accumulate threads nobody opened.
    """
    before = client.get("/api/v1/conversations", headers=auth_headers).json()["total"]

    body = ask(client, auth_headers).json()

    assert body["conversation_id"] is None
    assert body["message_id"] is None
    after = client.get("/api/v1/conversations", headers=auth_headers).json()["total"]
    assert after == before


def test_threads_are_listed_most_recently_used_first(
    client, auth_headers, document, fake_embedder, chat
):
    """
    Ordered by updated_at, which only moves if answering a question
    touches the conversation row. Adding a child message does not, so
    without an explicit touch the sidebar freezes at creation order —
    and the thread you just used sinks to the bottom.
    """
    first = start(client, auth_headers).json()["id"]
    second = start(client, auth_headers).json()["id"]

    ask(client, auth_headers, first, question="when is the invoice due?")

    listed = client.get("/api/v1/conversations", headers=auth_headers).json()
    assert [c["id"] for c in listed["conversations"]][:2] == [first, second]


def test_activity_keeps_moving_a_thread_to_the_top(
    client, auth_headers, document, fake_embedder, chat
):
    """
    On the first question the title is filled in, which updates the
    conversation row and makes onupdate fire by accident. Every turn
    after that changes nothing on the row, so without an explicit touch
    a thread you have been using all afternoon stops moving in the
    sidebar after its first message — and the bug hides behind a test
    that only ever asks once.
    """
    chat(FakeChatClient(["An invoice [1].", "rewritten", "Net 14 days [1]."]))
    conversation_id = start(client, auth_headers).json()["id"]

    ask(client, auth_headers, conversation_id, question="what did I save?")
    after_first = client.get(
        f"/api/v1/conversations/{conversation_id}", headers=auth_headers
    ).json()["updated_at"]

    ask(client, auth_headers, conversation_id, question="when is it due?")
    after_second = client.get(
        f"/api/v1/conversations/{conversation_id}", headers=auth_headers
    ).json()["updated_at"]

    assert after_second > after_first, "a later turn did not mark the thread as used"


def test_a_thread_can_be_renamed(client, auth_headers):
    conversation_id = start(client, auth_headers).json()["id"]

    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        headers=auth_headers,
        json={"title": "Coffee invoices"},
    )

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Coffee invoices"


def test_deleting_a_thread_keeps_the_documents(
    client, auth_headers, document, fake_embedder, chat, db_session
):
    """
    The obvious worry with "delete this conversation": whether it takes
    the sources with it. Messages hold a JSON snapshot rather than a
    foreign key, so there is nothing pointing at a document to cascade
    through — but that is exactly the kind of thing a later refactor
    breaks silently.
    """
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id)

    removed = client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)

    assert removed.status_code == 204
    assert (
        client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).status_code
        == 404
    )
    db_session.expire_all()
    assert db_session.get(Document, document.id) is not None, "the invoice was deleted"


def test_a_follow_up_is_condensed_before_it_is_searched(
    client, auth_headers, document, fake_embedder, chat
):
    """
    The single thing that makes a conversation work.
    """
    recorder = chat(
        FakeChatClient(
            [
                "An invoice from Kyiv Coffee Roasters [1].",  # first answer
                "when is the Kyiv Coffee Roasters invoice due?",  # condensation
                "Net 14 days from the invoice date [1].",  # second answer
            ]
        )
    )
    conversation_id = start(client, auth_headers).json()["id"]

    ask(client, auth_headers, conversation_id, question="what did I save about coffee?")
    second = ask(client, auth_headers, conversation_id, question="when is it due?").json()

    assert second["search_query"] == "when is the Kyiv Coffee Roasters invoice due?"
    assert second["question"] == "when is it due?", "the question shown is the one asked"
    # Three calls: answer, condense, answer. The first question skipped
    # condensation because it had no history to condense against.
    assert len(recorder.calls) == 3


def test_the_history_reaches_the_answering_prompt(
    client, auth_headers, document, fake_embedder, chat
):
    recorder = chat(
        FakeChatClient(
            [
                "An invoice from Kyiv Coffee Roasters [1].",
                "when is the invoice due?",
                "Net 14 days [1].",
            ]
        )
    )
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id, question="what did I save about coffee?")
    ask(client, auth_headers, conversation_id, question="when is it due?")

    answering_prompt = recorder.calls[2]["user"]
    assert "Conversation so far:" in answering_prompt
    assert "what did I save about coffee?" in answering_prompt


def test_stored_turns_are_replayed_in_the_order_they_happened(
    client, auth_headers, document, fake_embedder, chat
):
    """
    Ordering as it comes OUT of the database, which is a different
    question from ordering as it goes into build_prompt.

    Replayed backwards, the assistant appears to answer before it was
    asked and the model resolves "it" against the wrong turn. Nothing
    errors; the answers just quietly stop making sense.
    """
    recorder = chat(
        FakeChatClient(
            [
                "ALPHA the first reply [1].",
                "rewritten standalone question",
                "Net 14 days [1].",
            ]
        )
    )
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id, question="ZULU the first question")
    ask(client, auth_headers, conversation_id, question="when is it due?")

    answering_prompt = recorder.calls[2]["user"]
    assert answering_prompt.index("ZULU") < answering_prompt.index("ALPHA"), (
        "the stored history was replayed out of order"
    )
    assert "User: ZULU the first question" in answering_prompt
    assert "Assistant: ALPHA the first reply" in answering_prompt


def test_the_model_is_told_not_to_treat_history_as_evidence(
    client, auth_headers, document, fake_embedder, chat
):
    """
    The risk conversations introduce to the citation guarantee: a model
    that can see its own previous answers may restate one as fact
    without citing anything. The system prompt has to forbid it.
    """
    recorder = chat(FakeChatClient("Net 14 days [1]."))
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id)

    system = recorder.calls[0]["system"]
    assert "NOT evidence" in system


def test_the_rewritten_query_is_stored_on_the_user_turn(
    client, auth_headers, document, fake_embedder, chat
):
    """
    Model output, so it can be wrong, and a bad rewrite is invisible in
    the answer. Keeping it beside the question is what makes a complaint
    of "it answered about the wrong thing" diagnosable a week later.
    """
    chat(
        FakeChatClient(
            [
                "An invoice [1].",
                "when is the Kyiv Coffee Roasters invoice due?",
                "Net 14 days [1].",
            ]
        )
    )
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id, question="what did I save?")
    ask(client, auth_headers, conversation_id, question="when is it due?")

    messages = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).json()[
        "messages"
    ]
    user_turns = [m for m in messages if m["role"] == "user"]

    assert user_turns[0]["search_query"] is None, "a first question is not rewritten"
    assert user_turns[1]["search_query"] == "when is the Kyiv Coffee Roasters invoice due?"


def test_reopening_a_thread_shows_the_citations_the_answer_was_built_from(
    client, auth_headers, document, fake_embedder, chat
):
    conversation_id = start(client, auth_headers).json()["id"]
    live = ask(client, auth_headers, conversation_id).json()

    stored = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).json()[
        "messages"
    ][1]

    assert stored["cited_markers"] == live["cited_markers"]
    assert stored["retrieval"] == live["retrieval"]


def test_the_snapshot_does_not_follow_the_document_when_it_changes(
    client, auth_headers, document, fake_embedder, chat, db_session
):
    """
    Rename the document and the old answer keeps the old title,
    because that is what the assistant was looking at when it said it.
    """
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id)

    document.title = "Renamed after the fact"
    db_session.commit()

    stored = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).json()[
        "messages"
    ][1]

    assert stored["retrieval"]["citations"][0]["title"] == "Invoice 2026-0042"


def test_a_user_turn_carries_no_retrieval(client, auth_headers, document, fake_embedder, chat):
    conversation_id = start(client, auth_headers).json()["id"]
    ask(client, auth_headers, conversation_id)

    messages = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers).json()[
        "messages"
    ]

    assert messages[0]["retrieval"] is None
    assert messages[0]["cited_markers"] == []
    assert messages[1]["retrieval"] is not None


def test_another_users_thread_is_404_not_403(
    client, auth_headers, stranger_headers, document, fake_embedder, chat
):
    """
    What someone asks their notes is at least as sensitive as the notes.
    A 403 would confirm the id is real.
    """
    mine = start(client, auth_headers).json()["id"]

    assert client.get(f"/api/v1/conversations/{mine}", headers=stranger_headers).status_code == 404
    assert (
        client.patch(
            f"/api/v1/conversations/{mine}",
            headers=stranger_headers,
            json={"title": "hijacked"},
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/api/v1/conversations/{mine}", headers=stranger_headers).status_code == 404
    )


def test_a_stranger_cannot_append_to_my_thread(
    client, auth_headers, stranger_headers, document, fake_embedder, chat
):
    """
    The nastier version: not reading the thread but writing into it, and
    reading my documents in the process.
    """
    mine = start(client, auth_headers).json()["id"]

    response = ask(client, stranger_headers, mine, question="what is in here?")

    assert response.status_code == 404
    detail = client.get(f"/api/v1/conversations/{mine}", headers=auth_headers).json()
    assert detail["message_count"] == 0


def test_the_list_only_shows_my_own(
    client, auth_headers, stranger_headers, document, fake_embedder, chat
):
    mine = start(client, auth_headers).json()["id"]
    theirs = start(client, stranger_headers).json()["id"]

    listed = client.get("/api/v1/conversations", headers=auth_headers).json()
    ids = [c["id"] for c in listed["conversations"]]

    assert mine in ids
    assert theirs not in ids


def test_authentication_is_required(client):
    assert client.get("/api/v1/conversations").status_code in (401, 403)
    assert client.post("/api/v1/conversations", json={}).status_code in (401, 403)


def test_paging_is_bounded(client, auth_headers):
    assert client.get("/api/v1/conversations?limit=0", headers=auth_headers).status_code == 422
    assert client.get("/api/v1/conversations?limit=1000", headers=auth_headers).status_code == 422
    assert client.get("/api/v1/conversations?offset=-1", headers=auth_headers).status_code == 422


def test_a_blank_title_is_refused(client, auth_headers):
    conversation_id = start(client, auth_headers).json()["id"]
    response = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        headers=auth_headers,
        json={"title": ""},
    )
    assert response.status_code == 422


def test_a_missing_thread_is_404(client, auth_headers):
    assert (
        client.get(f"/api/v1/conversations/{uuid.uuid4()}", headers=auth_headers).status_code == 404
    )
