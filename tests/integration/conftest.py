import pytest

from app.core.db import engine


@pytest.fixture(autouse=True)
async def _dispose_async_engine():
    yield
    await engine.dispose()

@pytest.fixture(scope="session")
def keep_rows(request) -> bool:
    return bool(request.config.getoption("--keep-rows"))