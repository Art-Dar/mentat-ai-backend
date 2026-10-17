import uuid

import pytest
from sqlalchemy import select

from app.main import app
from app.models import Chunk, Document, DocumentSource, IngestionStatus, User
from app.models.chunk import EMBEDDING_DIM
from app.services.ingestion.embedder import get_embedder
from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.llm.base import LlmError, LlmUnavailable
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
    """
    Installs a chat client and hands it back so a test can reconfigure
    it. Default answer cites [1], which is the normal case.
    """
    holder: dict = {"client": FakeChatClient("Net 14 days from the invoice date [1].")}
    app.dependency_overrides[get_chat_client] = lambda: holder["client"]

    def use(client: FakeChatClient) -> FakeChatClient:
        holder["client"] = client
        return client

    use.current = lambda: holder["client"]
    yield use
    app.dependency_overrides.pop(get_chat_client, None)


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def make_chunk(db_session, seeded_user):
    created_documents: list[uuid.UUID] = []
    created_users: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(content: str = INVOICE, *, owner: User | None = None, **fields) -> Document:
        if owner is None:
            owner = db_session.execute(select(User).where(User.email == email)).scalars().one()

        document = Document(
            id=uuid.uuid4(),
            user_id=owner.id,
            source=fields.pop("source", DocumentSource.PDF),
            status=IngestionStatus.COMPLETED,
            content=content,
            title=fields.pop("title", "Invoice 2026-0042"),
            page_map=fields.pop(
                "page_map",
                [
                    {
                        "index": 0,
                        "start_char": 0,
                        "end_char": len(content),
                        "needs_ocr": False,
                        "source": "pdf",
                    }
                ],
            ),
            **fields,
        )
        db_session.add(document)
        db_session.flush()
        created_documents.append(document.id)
        db_session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                content=content,
                start_char=0,
                end_char=len(content),
                embedding=[0.01] * EMBEDDING_DIM,
                embedding_model="fake",
            )
        )
        db_session.commit()
        return document

    def stranger() -> User:
        other = User(id=uuid.uuid4(), email=f"other-{uuid.uuid4().hex[:8]}@example.com")
        db_session.add(other)
        db_session.commit()
        created_users.append(other.id)
        return other

    build.stranger = stranger
    yield build

    for document_id in created_documents:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    db_session.commit()
    for user_id in created_users:
        user = db_session.get(User, user_id)
        if user is not None:
            db_session.delete(user)
    db_session.commit()


def ask(client, headers, **body):
    body.setdefault("question", "when is the invoice due?")
    body.setdefault("min_similarity", -1.0)
    return client.post("/api/v1/ask", headers=headers, json=body)


def test_a_question_comes_back_answered_and_cited(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    make_chunk()
    body = ask(client, auth_headers).json()

    assert body["answer"] == "Net 14 days from the invoice date [1]."
    assert body["cited_markers"] == ["[1]"]
    assert body["grounded"] is True
    assert body["retrieval"]["chunk_count"] == 1
    assert body["retrieval"]["citations"][0]["marker"] == "[1]"


def test_the_sources_actually_reach_the_model(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    """
    The test that would catch retrieval being wired up but never passed
    on. A fake that answers regardless of its prompt cannot tell a
    prompt carrying the invoice from an empty one, so the assertion has
    to be on what the model was sent.
    """
    make_chunk()
    recorder = chat(FakeChatClient("Net 14 days [1]."))

    ask(client, auth_headers)

    prompt = recorder.calls[0]["user"]
    assert "net 14 days" in prompt.lower(), "the chunk never reached the prompt"
    assert "<<<SOURCE [1]>>>" in prompt, "sources are not delimited"
    assert "Invoice 2026-0042" in prompt, "the citation's title is missing"


def test_a_marker_the_model_invented_never_reaches_the_client(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    """
    The guarantee the frontend contract is allowed to rely on. A
    footnote marker that resolves to nothing looks exactly like one that
    resolves, right up until the user clicks it.
    """
    make_chunk()
    chat(FakeChatClient("Net 14 days [1]. The vendor is Kyiv Coffee Roasters [7]."))

    body = ask(client, auth_headers).json()

    assert "[7]" not in body["answer"]
    assert body["answer"] == "Net 14 days [1]. The vendor is Kyiv Coffee Roasters."
    assert body["cited_markers"] == ["[1]"]


def test_an_answer_that_cites_nothing_is_reported_as_ungrounded(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    """
    Not an error. "I could not find that" is the correct answer to an
    unanswerable question — but the client must know not to present it
    as verifiable.
    """
    make_chunk()
    chat(FakeChatClient("I could not find that in your documents."))

    body = ask(client, auth_headers).json()

    assert body["grounded"] is False
    assert body["cited_markers"] == []
    # The sources are still returned: they were consulted, and the user
    # may want to look at what the assistant decided was irrelevant.
    assert body["retrieval"]["chunk_count"] == 1


def test_an_empty_knowledge_base_is_answered_not_errored(client, auth_headers, fake_embedder, chat):
    chat(FakeChatClient("I could not find that in your documents."))

    response = ask(client, auth_headers, min_similarity=0.999999)

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["retrieval"]["citations"] == []


def test_the_search_query_is_the_question_when_there_is_no_history(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    make_chunk()
    recorder = chat(FakeChatClient("Net 14 days [1]."))

    body = ask(client, auth_headers, question="when is the invoice due?").json()

    assert body["search_query"] == "when is the invoice due?"
    assert len(recorder.calls) == 1, "a standalone question needs no condensation call"


def test_another_users_documents_never_reach_the_answer(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    """
    The same guarantee /search has, re-asserted at the point where a
    leak would be laundered through prose. A chunk that reaches the
    prompt has already leaked even if the model never quotes it.
    """
    make_chunk("my own notes about espresso grind size and dose")
    make_chunk(
        "SOMEBODY ELSE'S PRIVATE RECEIPT, total 349.50 UAH",
        owner=make_chunk.stranger(),
    )
    recorder = chat(FakeChatClient("Grind size matters [1]."))

    body = ask(client, auth_headers, top_k=50).json()

    assert "PRIVATE RECEIPT" not in recorder.calls[0]["user"]
    assert body["retrieval"]["document_count"] == 1
    for citation in body["retrieval"]["citations"]:
        assert "PRIVATE RECEIPT" not in citation["quote"]


def test_authentication_is_required(client, fake_embedder, chat):
    assert client.post("/api/v1/ask", json={"question": "anything"}).status_code in (401, 403)


def test_the_model_being_down_is_a_503(client, auth_headers, make_chunk, fake_embedder, chat):
    """503 invites a retry, which is the right advice for a rate limit."""
    make_chunk()
    chat(FakeChatClient(error=LlmUnavailable("groq unavailable after 3 attempts")))

    response = ask(client, auth_headers)

    assert response.status_code == 503
    assert "try again" in response.json()["error"]["message"].lower()


def test_a_misconfigured_model_is_a_500_not_a_503(
    client, auth_headers, make_chunk, fake_embedder, chat
):
    """
    A missing API key will be just as missing on the next attempt. A 503
    here would send clients into a retry loop against a fault only a
    deploy can fix.
    """
    make_chunk()
    chat(FakeChatClient(error=LlmError("GROQ_API_KEY is not set")))

    response = ask(client, auth_headers)

    assert response.status_code == 500
    # The user is told it is broken, not which secret is missing.
    assert "GROQ_API_KEY" not in response.text


def test_the_model_is_never_called_before_authorization(client, chat, fake_embedder):
    """
    A request from nobody must not cost a token. Dependencies resolve
    before the handler, so this also pins that the auth dependency runs
    first.
    """
    recorder = chat(FakeChatClient("should not be reached"))

    client.post("/api/v1/ask", json={"question": "anything"})

    assert recorder.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", ""),
        ("question", "x" * 1001),
        ("top_k", 0),
        ("top_k", 101),
        ("min_similarity", -1.1),
        ("min_similarity", 1.1),
        ("conversation_id", "not-a-uuid"),
    ],
)
def test_a_malformed_request_is_422(client, auth_headers, fake_embedder, chat, field, value):
    body = {"question": "a real question"}
    body[field] = value
    assert client.post("/api/v1/ask", headers=auth_headers, json=body).status_code == 422


def test_an_unknown_field_is_refused(client, auth_headers, fake_embedder, chat):
    response = client.post(
        "/api/v1/ask",
        headers=auth_headers,
        json={"question": "espresso", "temperature": 0.9},
    )
    assert response.status_code == 422, "a client must not be able to set generation knobs"


def test_a_conversation_that_is_not_yours_is_404(client, auth_headers, fake_embedder, chat):
    response = ask(client, auth_headers, conversation_id=str(uuid.uuid4()))
    assert response.status_code == 404
