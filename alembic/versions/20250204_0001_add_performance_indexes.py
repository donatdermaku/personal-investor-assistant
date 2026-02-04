"""add performance indexes

Revision ID: 20250204_0001
Revises: 20250124_0003
Create Date: 2026-02-04 23:41:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20250204_0001'
down_revision = '20250124_0003'
branch_labels = None
depends_on = None


def upgrade():
    # Trades indexes for portfolio queries
    op.create_index(
        'idx_trades_portfolio_date_ticker',
        'trades',
        ['portfolio_id', 'date', 'ticker'],
        unique=False
    )
    op.create_index(
        'idx_trades_date',
        'trades',
        ['date'],
        unique=False
    )
    
    # Runs index for latest run queries
    op.create_index(
        'idx_runs_created_at',  
        'runs',
        [sa.text('created_at DESC')],
        unique=False,
        postgresql_using='btree'
    )
    
    # Cache freshness queries
    op.create_index(
        'idx_cache_updated_at',
        'data_cache_index',
        [sa.text('updated_at DESC')],
        unique=False,
        postgresql_using='btree'
    )
    op.create_index(
        'idx_cache_status_source',
        'data_cache_index',
        ['status', 'source'],
        unique=False
    )


def downgrade():
    op.drop_index('idx_cache_status_source', table_name='data_cache_index')
    op.drop_index('idx_cache_updated_at', table_name='data_cache_index')
    op.drop_index('idx_runs_created_at', table_name='runs')
    op.drop_index('idx_trades_date', table_name='trades')
    op.drop_index('idx_trades_portfolio_date_ticker', table_name='trades')
