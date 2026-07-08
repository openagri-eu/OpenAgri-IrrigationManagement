"""Add UUID id to soil_type_values and crop_kc

Revision ID: 09ba0c74e019
Revises: 7c78a3ccee81
Create Date: 2026-07-07 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '09ba0c74e019'
down_revision: Union[str, None] = '7c78a3ccee81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS pgcrypto')

    # --- soil_type_values ---
    op.add_column('soil_type_values', sa.Column('id', postgresql.UUID(as_uuid=True), nullable=True))
    op.execute('UPDATE soil_type_values SET id = gen_random_uuid()')
    op.alter_column('soil_type_values', 'id', nullable=False,
                     server_default=sa.text('gen_random_uuid()'))
    op.drop_constraint('soil_type_values_pkey', 'soil_type_values', type_='primary')
    op.create_primary_key('soil_type_values_pkey', 'soil_type_values', ['id'])
    op.create_unique_constraint('uq_soil_type_values_soil_type', 'soil_type_values', ['soil_type'])

    # --- crop_kc ---
    op.add_column('crop_kc', sa.Column('id', postgresql.UUID(as_uuid=True), nullable=True))
    op.execute('UPDATE crop_kc SET id = gen_random_uuid()')
    op.alter_column('crop_kc', 'id', nullable=False,
                     server_default=sa.text('gen_random_uuid()'))
    op.drop_constraint('crop_kc_pkey', 'crop_kc', type_='primary')
    op.create_primary_key('crop_kc_pkey', 'crop_kc', ['id'])
    op.create_unique_constraint('uq_crop_kc_crop', 'crop_kc', ['crop'])


def downgrade() -> None:
    op.drop_constraint('uq_crop_kc_crop', 'crop_kc', type_='unique')
    op.drop_constraint('crop_kc_pkey', 'crop_kc', type_='primary')
    op.create_primary_key('crop_kc_pkey', 'crop_kc', ['crop'])
    op.drop_column('crop_kc', 'id')

    op.drop_constraint('uq_soil_type_values_soil_type', 'soil_type_values', type_='unique')
    op.drop_constraint('soil_type_values_pkey', 'soil_type_values', type_='primary')
    op.create_primary_key('soil_type_values_pkey', 'soil_type_values', ['soil_type'])
    op.drop_column('soil_type_values', 'id')
