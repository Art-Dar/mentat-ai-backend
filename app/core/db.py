import sys
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from app.core.config import settings

# asyncpg connections belong to the event loop that opened them, and the
# test suite has two kinds of loop it cannot reconcile: anyio builds a
# fresh one for every async test, and Starlette's TestClient runs the app
# in one of its own. A POOLED connection handed across that boundary
# raises "Event loop is closed" or "another operation is in progress" — in
# whichever test happens to touch it next, which is why those failures
# land nowhere near their cause.
#
# NullPool opens and closes a connection per checkout, so nothing survives
# to be reused in the wrong loop. Test-only: in production this would cost
# a TCP handshake plus authentication on every single request.
_UNDER_PYTEST = "pytest" in sys.modules

# The two branches must stay separate: pool_size passed together with
# NullPool is a TypeError at engine construction, not a warning.
_POOL_KWARGS: dict = (
    {"poolclass": NullPool}
    if _UNDER_PYTEST
    else {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}
)

# ONE engine, and it must be created BEFORE the sessionmaker below.
# async_sessionmaker captures the engine OBJECT at call time, so creating a
# second engine further down the module would rebind the name `engine`
# while AsyncSessionLocal kept using the first one — the pool settings here
# would silently have no effect on any session the app opens.
engine = create_async_engine(settings.database_url, echo=settings.debug, **_POOL_KWARGS)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session