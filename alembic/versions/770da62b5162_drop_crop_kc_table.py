"""Drop crop_kc table

Revision ID: 770da62b5162
Revises: 09ba0c74e019
Create Date: 2026-09-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '770da62b5162'
down_revision: Union[str, None] = '09ba0c74e019'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('crop_kc')


def downgrade() -> None:
    op.create_table(
        'crop_kc',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('crop', sa.String(), nullable=False),
        sa.Column('kc_init', sa.Float(), nullable=False),
        sa.Column('kc_mid', sa.Float(), nullable=False),
        sa.Column('kc_end', sa.Float(), nullable=False),
        sa.UniqueConstraint('crop', name='uq_crop_kc_crop'),
    )
