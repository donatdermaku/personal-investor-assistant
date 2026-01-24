from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Index, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, default="Main Portfolio")
    base_currency = Column(String, nullable=False, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

    runs = relationship("Run", back_populates="portfolio")
    snapshots = relationship("HoldingsSnapshot", back_populates="portfolio")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True, nullable=False)
    date = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    action = Column(String, nullable=False)
    quantity = Column(Float, nullable=False, default=0.0)
    price = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)
    fees = Column(Float, nullable=True)
    currency = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class HoldingsSnapshot(Base):
    __tablename__ = "holdings_snapshots"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True, nullable=False)
    as_of_date = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False, default=0.0)
    cost_basis = Column(Float, nullable=True)

    portfolio = relationship("Portfolio", back_populates="snapshots")


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True, nullable=False)
    run_type = Column(String, nullable=True)
    status = Column(String, nullable=False, default="running")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    manifest_json = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    message = Column(Text, nullable=True)

    portfolio = relationship("Portfolio", back_populates="runs")
    artifacts = relationship("RunArtifact", back_populates="run")


class RunArtifact(Base):
    __tablename__ = "run_artifacts"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True, nullable=False)
    artifact_key = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="artifacts")


Index("idx_runs_portfolio_created", Run.portfolio_id, Run.created_at.desc())
Index("idx_transactions_portfolio_date", Transaction.portfolio_id, Transaction.date)
Index("idx_run_artifacts_run_key", RunArtifact.run_id, RunArtifact.artifact_key)
Index("idx_holdings_snapshots_portfolio_date", HoldingsSnapshot.portfolio_id, HoldingsSnapshot.as_of_date)


class DataCacheIndex(Base):
    __tablename__ = "data_cache_index"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False, index=True)
    key = Column(String, nullable=False, index=True)
    asof_date = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    ttl_seconds = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="fresh")
    coverage_pct = Column(Float, nullable=True)
    storage_path = Column(String, nullable=False)
    error_code = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("source", "key", name="uq_cache_source_key"),)
