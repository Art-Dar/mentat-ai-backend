"""Documents table created

Revision ID: cb8c9462dbed
Revises: 57bf0e3454ad
Create Date: 2026-10-03 01:09:07.320347

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cb8c9462dbed'
down_revision: Union[str, Sequence[str], None] = '57bf0e3454ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
