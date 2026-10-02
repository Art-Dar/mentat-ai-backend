"""Users table expanded, auth provider for auth way

Revision ID: 57bf0e3454ad
Revises: 8dcfb666f98d
Create Date: 2026-10-02 22:41:09.655990

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '57bf0e3454ad'
down_revision: Union[str, Sequence[str], None] = '8dcfb666f98d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
