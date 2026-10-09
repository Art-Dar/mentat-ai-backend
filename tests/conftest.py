import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
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


def pytest_addoption(parser):
    parser.addoption(
        "--keep-rows",
        action="store_true",
        default=False,
        help="Leave integration-test rows in the database for inspection.",
    )