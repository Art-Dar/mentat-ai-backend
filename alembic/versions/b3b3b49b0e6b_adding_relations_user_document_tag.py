"""adding relations user-document-tag

Revision ID: b3b3b49b0e6b
Revises: 151f526dea78
Create Date: 2026-10-03 22:18:57.704803

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3b3b49b0e6b'
down_revision: Union[str, Sequence[str], None] = '151f526dea78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
