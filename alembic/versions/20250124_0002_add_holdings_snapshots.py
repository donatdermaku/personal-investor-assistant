from alembic import op
import sqlalchemy as sa


revision = "20250124_0002"
down_revision = "20250124_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "holdings_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("portfolio_id", sa.Integer(), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("as_of_date", sa.DateTime(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("shares", sa.Float(), nullable=False),
        sa.Column("cost_basis", sa.Float(), nullable=True),
    )
    op.create_index(
        "idx_holdings_snapshots_portfolio_date",
        "holdings_snapshots",
        ["portfolio_id", "as_of_date"],
    )


def downgrade() -> None:
    op.drop_index("idx_holdings_snapshots_portfolio_date", table_name="holdings_snapshots")
    op.drop_table("holdings_snapshots")
