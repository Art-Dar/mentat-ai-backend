"""
Seed the two Mentat users.
...
"""

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()

from app.core.db import AsyncSessionLocal  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models.user import User  # noqa: E402

USERS = [
    (os.environ["SEED_USER_1_USERNAME"], os.environ["SEED_USER_1_PASSWORD"]),
    (os.environ["SEED_USER_2_USERNAME"], os.environ["SEED_USER_2_PASSWORD"]),
]


async def seed() -> None:
    async with AsyncSessionLocal() as db:
        for username, password in USERS:
            existing = await db.execute(select(User).where(User.username == username))
            if existing.scalar_one_or_none() is not None:
                print(f"skip: {username} already exists")
                continue
            db.add(User(username=username, hashed_password=hash_password(password)))
            print(f"created: {username}")
        await db.commit()


if __name__ == "__main__":
    asyncio.run(seed())
