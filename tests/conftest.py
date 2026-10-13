import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.main import app
from app.models.user import AuthIdentity, AuthProvider, User

# Plain sync engine, used only by test fixtures for setup/teardown.
# The app itself always talks to the DB through the async engine inside
# whatever event loop TestClient spins up per request — mixing that async
# engine with asyncio.run() in a fixture causes cross-event-loop asyncpg
# errors ("another operation is in progress"). A sync engine sidesteps
# event loops entirely for fixture-side data setup.
_sync_engine = create_engine(settings.database_url.replace("+asyncpg", "+psycopg2"))


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    with Session(_sync_engine) as session:
        yield session


@pytest.fixture
def seeded_user(db_session):
    """Creates a user with a unique email so tests don't collide on re-runs."""
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    password = "correct-horse-battery-staple"

    user = User(email=email, email_verified=True)
    user.identities.append(
        AuthIdentity(
            provider=AuthProvider.PASSWORD,
            provider_user_id=email,
            password_hash=hash_password(password),
        )
    )
    db_session.add(user)
    db_session.commit()

    yield email, password

    db_session.delete(user)
    db_session.commit()


@pytest.fixture
def register_email(db_session):
    """A unique email for /auth/register tests; the user it creates, if any,
    is removed afterwards."""
    email = f"register-{uuid.uuid4().hex[:8]}@example.com"

    yield email

    user = db_session.scalar(select(User).where(User.email == email))
    if user is not None:
        db_session.delete(user)
        db_session.commit()


@pytest.fixture
def seeded_google_user(db_session):
    """A user already linked to a Google identity, for /auth/google tests."""
    email = f"google-{uuid.uuid4().hex[:8]}@example.com"
    google_sub = f"google-sub-{uuid.uuid4().hex[:8]}"

    user = User(email=email, email_verified=True)
    user.identities.append(
        AuthIdentity(
            provider=AuthProvider.GOOGLE,
            provider_user_id=google_sub,
            password_hash=None,
        )
    )
    db_session.add(user)
    db_session.commit()

    yield email, google_sub

    db_session.delete(user)
    db_session.commit()


def pytest_addoption(parser):
    parser.addoption(
        "--keep-rows",
        action="store_true",
        default=False,
        help="Leave integration-test rows in the database for inspection.",
    )


@pytest.fixture(autouse=True)
def pipeline_isolation(request, monkeypatch):
    """
    Starlette's TestClient runs background tasks before `client.post`
    returns, so without this every /ingest test would run a full pipeline —
    and call the real Voyage API from CI, with a dummy key, over the
    network. Tests marked `runs_pipeline` get a FakeEmbedder instead, so
    they exercise the real wiring without leaving the machine.
    """
    if request.node.get_closest_marker("runs_pipeline"):
        from app.services.ingestion.fake_embedder import FakeEmbedder

        monkeypatch.setattr("app.services.ingestion.pipeline.get_embedder", lambda: FakeEmbedder())
        yield None
        return

    scheduled = []

    async def record(document_id, background_tasks=None):
        # The BackgroundTasks instance is recorded too, so a test can prove
        # the endpoint handed off instead of awaiting the pipeline inline.
        scheduled.append((document_id, background_tasks))

    monkeypatch.setattr("app.api.v1.ingest.schedule_processing", record)
    yield scheduled
