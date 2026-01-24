"""add data_cache_index table

Revision ID: 20250124_0003
Revises: 20250124_0002
Create Date: 2026-01-24
"""

from alembic import op
import sqlalchemy as sa


revision = "20250124_0003"
down_revision = "20250124_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_cache_index",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("asof_date", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="fresh"),
        sa.Column("coverage_pct", sa.Float(), nullable=True),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_unique_constraint("uq_cache_source_key", "data_cache_index", ["source", "key"])
    op.create_index("idx_cache_source_key", "data_cache_index", ["source", "key"])


def downgrade() -> None:
    op.drop_index("idx_cache_source_key", table_name="data_cache_index")
    op.drop_constraint("uq_cache_source_key", "data_cache_index", type_="unique")
    op.drop_table("data_cache_index")
