from alembic import op
import sqlalchemy as sa


revision = "20250124_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("base_currency", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("portfolio_id", sa.Integer(), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("amount", sa.Float(), nullable=True),
        sa.Column("fees", sa.Float(), nullable=True),
        sa.Column("currency", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("portfolio_id", sa.Integer(), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("run_type", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("manifest_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
    )
    op.create_table(
        "run_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("artifact_key", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_runs_portfolio_created", "runs", ["portfolio_id", "created_at"])
    op.create_index("idx_transactions_portfolio_date", "transactions", ["portfolio_id", "date"])
    op.create_index("idx_run_artifacts_run_key", "run_artifacts", ["run_id", "artifact_key"])


def downgrade() -> None:
    op.drop_index("idx_run_artifacts_run_key", table_name="run_artifacts")
    op.drop_index("idx_transactions_portfolio_date", table_name="transactions")
    op.drop_index("idx_runs_portfolio_created", table_name="runs")
    op.drop_table("run_artifacts")
    op.drop_table("runs")
    op.drop_table("transactions")
    op.drop_table("portfolios")
