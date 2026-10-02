"""Chunks table created

Revision ID: 52d9bd6aec93
Revises: cb8c9462dbed
Create Date: 2026-10-03 01:28:06.192611

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52d9bd6aec93'
down_revision: Union[str, Sequence[str], None] = 'cb8c9462dbed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
