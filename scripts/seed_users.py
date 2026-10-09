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
from app.models.user import AuthIdentity, AuthProvider, User  # noqa: E402

USERS = [
    (os.environ["SEED_USER_1_EMAIL"], os.environ["SEED_USER_1_PASSWORD"]),
    (os.environ["SEED_USER_2_EMAIL"], os.environ["SEED_USER_2_PASSWORD"]),
]


USERS = [
    (os.environ["SEED_USER_1_EMAIL"], os.environ["SEED_USER_1_PASSWORD"]),
    (os.environ["SEED_USER_2_EMAIL"], os.environ["SEED_USER_2_PASSWORD"]),
]


async def seed() -> None:
    async with AsyncSessionLocal() as db:
        for email, password in USERS:
            existing = await db.execute(select(User).where(User.email == email))
            if existing.scalar_one_or_none() is not None:
                print(f"skip: {email} already exists")
                continue

            user = User(email=email, email_verified=True)
            # for password auth that is the email itself
            # for Google it will be the opaque `sub` claim
            user.identities.append(
                AuthIdentity(
                    provider=AuthProvider.PASSWORD,
                    provider_user_id=email,
                    password_hash=hash_password(password),
                )
            )
            db.add(user)
            print(f"created: {email}")

        await db.commit()


if __name__ == "__main__":
    asyncio.run(seed())