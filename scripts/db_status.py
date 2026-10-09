"""
What is actually in the database, read through the app's own settings.

Useful when a GUI client and the application disagree about what exists:
this connects with exactly the URL, driver and credentials the app uses,
so there is no second connection to be wrong about.

    poetry run python -m scripts.db_status
"""

import asyncio
import re

from sqlalchemy import text

from app.core.config import settings
from app.core.db import AsyncSessionLocal

# Queries are plain SQL rather than ORM so this keeps working even when a
# model and the migrated schema have drifted apart — which is one of the
# situations you would run this script to diagnose.
COUNTS = """
SELECT
  (SELECT count(*) FROM users)     AS users,
  (SELECT count(*) FROM documents) AS documents,
  (SELECT count(*) FROM chunks)    AS chunks
"""

PER_DOCUMENT = """
SELECT d.id::text, d.status, coalesce(d.language, '?') AS lang,
       count(c.id) AS chunks, count(c.embedding) AS embedded,
       coalesce(left(d.title, 28), '(no title)') AS title
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.id
ORDER BY d.created_at DESC
LIMIT 15
"""

PROBLEMS = """
SELECT count(*) FROM chunks
WHERE embedding IS NULL OR vector_dims(embedding) <> :dim OR embedding_model IS NULL
"""

INDEX = """
SELECT count(*) FROM pg_indexes
WHERE tablename = 'chunks' AND indexdef ILIKE '%hnsw%vector_cosine_ops%'
"""


def masked(url: str) -> str:
    """Never print the password, even to a local terminal."""
    return re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", url)


async def main() -> None:
    print(f"connected to   {masked(settings.database_url)}")
    print(f"model          {settings.embedding_model}")

    async with AsyncSessionLocal() as db:
        users, documents, chunks = (await db.execute(text(COUNTS))).one()
        print(f"\nusers {users}   documents {documents}   chunks {chunks}")

        if not documents:
            print(
                "\nNothing ingested yet. Either nothing has written to this database,\n"
                "or the integration tests cleaned up after themselves — they only keep\n"
                "their rows with: pytest tests/integration --keep-rows"
            )
            return

        print(f"\n{'document':38} {'status':11} {'lang':5} {'chunks':>6} {'emb':>4}  title")
        for doc_id, status, lang, n, embedded, title in (
            await db.execute(text(PER_DOCUMENT))
        ).all():
            print(f"{doc_id:38} {status:11} {lang:5} {n:>6} {embedded:>4}  {title}")

        from app.models.chunk import EMBEDDING_DIM

        bad = (await db.execute(text(PROBLEMS), {"dim": EMBEDDING_DIM})).scalar_one()
        has_index = (await db.execute(text(INDEX))).scalar_one()

        print(f"\nbad chunks     {bad} (NULL, wrong width, or no model recorded)")
        print(f"hnsw index     {'present' if has_index else 'MISSING — search will seq scan'}")


if __name__ == "__main__":
    asyncio.run(main())
