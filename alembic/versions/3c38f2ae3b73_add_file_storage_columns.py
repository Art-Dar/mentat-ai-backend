"""add file storage columns

Revision ID: 3c38f2ae3b73
Revises: b3b3b49b0e6b
Create Date: 2026-10-14 19:59:04.896768

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '3c38f2ae3b73'
down_revision: Union[str, Sequence[str], None] = 'b3b3b49b0e6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SOURCE_ENUM = "document_source"


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE is transactional from PostgreSQL 12 onward,
    # so this is safe inside Alembic's transaction. The catch: the new
    # value cannot be USED in the same transaction that adds it. Nothing
    # here writes a 'pdf' row, so this is fine — but a data migration that
    # backfills rows with it must live in a SEPARATE revision.
    op.execute(f"ALTER TYPE {_SOURCE_ENUM} ADD VALUE IF NOT EXISTS 'pdf'")

    # Nullable throughout: every document already in the table arrived as
    # text and has no file. A NOT NULL column here would need a default
    # that lies about where those documents came from.
    op.add_column("documents", sa.Column("storage_key", sa.String(512), nullable=True))
    op.add_column("documents", sa.Column("file_size", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("original_filename", sa.String(255), nullable=True))

    # JSONB, not JSON: it is stored decomposed, so it is read without
    # reparsing and can be indexed later if the OCR job needs to find
    # pages across documents.
    op.add_column(
        "documents",
        sa.Column("page_map", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    # Partial index: only rows that actually have a file. The predicate
    # keeps it small, and it is the index the orphan sweeper will use to
    # reconcile the bucket against the table.
    op.create_index(
        "ix_documents_storage_key",
        "documents",
        ["storage_key"],
        unique=True,
        postgresql_where=sa.text("storage_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_storage_key", table_name="documents")
    op.drop_column("documents", "page_map")
    op.drop_column("documents", "original_filename")
    op.drop_column("documents", "file_size")
    op.drop_column("documents", "storage_key")
    # The enum value is deliberately NOT removed. PostgreSQL has no
    # ALTER TYPE ... DROP VALUE; undoing it means recreating the type and
    # rewriting every column that uses it. An unused enum member is
    # harmless, and a downgrade that rewrites the documents table is not.
