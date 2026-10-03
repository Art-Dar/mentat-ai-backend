"""enable pgvector extension

Revision ID: d9704a3ac199
Revises: 
Create Date: 2026-10-03 15:09:15.574649

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd9704a3ac199'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")