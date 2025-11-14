"""Add soil analysis v2 models

Revision ID: 002_soil_analysis_v2
Revises: 232f37fc06a1
Create Date: 2025-11-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_soil_analysis_v2'
down_revision: Union[str, None] = '232f37fc06a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create AnalysisStatus enum type
    analysis_status_enum = sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', name='analysisstatus')
    analysis_status_enum.create(op.get_bind(), checkfirst=True)

    # Create soil table
    op.create_table('soil',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('field_capacity', sa.Float(), nullable=False),
        sa.Column('wilting_point', sa.Float(), nullable=False),
        sa.Column('et0_coefficient', sa.Float(), nullable=False, server_default='0.174'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Add columns to dataset table
    op.add_column('dataset', sa.Column('name', sa.String(), nullable=False))
    op.add_column('dataset', sa.Column('soil_id', sa.Integer(), nullable=True))
    op.add_column('dataset', sa.Column('uploaded_at', sa.DateTime(), nullable=True, server_default=sa.func.now()))
    op.add_column('dataset', sa.Column('analysis_status', sa.String(length=20), nullable=True, server_default='PENDING'))

    # Add foreign key constraint to dataset.soil_id
    op.create_foreign_key('fk_dataset_soil_id', 'dataset', 'soil', ['soil_id'], ['id'], ondelete='SET NULL')
    
    # Create soil_analysis_timeseries table
    op.create_table('soil_analysis_timeseries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('avg_soil_moisture', sa.Float(), nullable=True),
        sa.Column('smi', sa.Float(), nullable=True),
        sa.Column('eto', sa.Float(), nullable=True),
        sa.Column('water_balance', sa.Float(), nullable=True),
        sa.Column('irrigation_need', sa.String(length=50), nullable=True),
        sa.Column('saturation_event', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('saturation_type', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['dataset_id'], ['dataset.id'], ondelete='CASCADE')
    )
    
    # Create soil_analysis_event table
    op.create_table('soil_analysis_event',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_name', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('first_occurrence', sa.Date(), nullable=True),
        sa.Column('last_occurrence', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        # sa.ForeignKeyConstraint(['dataset_name'], ['dataset.name'], ondelete='CASCADE')
    )

    # Create indexes
    op.create_index('ix_soil_analysis_timeseries_dataset_date', 'soil_analysis_timeseries', ['dataset_id', 'date'])
    op.create_index('ix_soil_analysis_event_dataset_type', 'soil_analysis_event', ['dataset_name', 'event_type'])
    op.create_index('ix_dataset_name', 'dataset', ['name'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_soil_analysis_event_dataset_type', table_name='soil_analysis_event')
    op.drop_index('ix_soil_analysis_timeseries_dataset_date', table_name='soil_analysis_timeseries')
    
    # Drop tables
    op.drop_table('soil_analysis_event')
    op.drop_table('soil_analysis_timeseries')
    
    # Remove foreign key and columns from dataset
    op.drop_constraint('fk_dataset_soil_id', 'dataset', type_='foreignkey')
    op.drop_column('dataset', 'analysis_status')
    op.drop_column('dataset', 'uploaded_at')
    op.drop_column('dataset', 'soil_id')
    op.drop_column('dataset', 'name')
    
    # Drop soil table
    op.drop_table('soil')

    # Drop enum type
    analysis_status_enum = sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', name='analysisstatus')
    analysis_status_enum.drop(op.get_bind(), checkfirst=True)
