import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.main import app
from app.models.user import User

# Plain sync engine, used only by test fixtures for setup/teardown.
# The app itself always talks to the DB through the async engine inside
# whatever event loop TestClient spins up per request — mixing that async
# engine with asyncio.run() in a fixture causes cross-event-loop asyncpg
# errors ("another operation is in progress"). A sync engine sidesteps
# event loops entirely for fixture-side data setup.
_sync_engine = create_engine(settings.database_url.replace("+asyncpg", ""))


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded_user() -> tuple[str, str]:
    """Creates a user with a unique username so tests don't collide on re-runs."""
    username = f"test_user_{uuid.uuid4().hex[:8]}"
    password = "test_password_123"

    with Session(_sync_engine) as db:
        db.add(User(username=username, hashed_password=hash_password(password)))
        db.commit()

    return username, password
