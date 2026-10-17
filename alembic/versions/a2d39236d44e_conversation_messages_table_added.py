"""conversation & messages table added

Revision ID: a2d39236d44e
Revises: 3c38f2ae3b73
Create Date: 2026-10-17 17:25:44.737469

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2d39236d44e'
down_revision: Union[str, Sequence[str], None] = '3c38f2ae3b73'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
