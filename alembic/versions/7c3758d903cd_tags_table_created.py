"""Tags table created

Revision ID: 7c3758d903cd
Revises: 52d9bd6aec93
Create Date: 2026-10-03 01:32:58.934617

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c3758d903cd'
down_revision: Union[str, Sequence[str], None] = '52d9bd6aec93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
