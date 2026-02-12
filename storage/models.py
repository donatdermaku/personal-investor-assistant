from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import JSON

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    portfolios = relationship("Portfolio", back_populates="owner")
    watchlist_items = relationship("WatchlistItem", back_populates="user")
    settings = relationship("AppSettings", back_populates="user")

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, default="Main Portfolio")
    currency = Column(String, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="portfolios")
    trades = relationship("Trade", back_populates="portfolio")
    snapshots = relationship("HoldingsSnapshot", back_populates="portfolio")
    runs = relationship("Run", back_populates="portfolio")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True)
    date = Column(DateTime, nullable=False) # Store as datetime, but meant as date
    ticker = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False) # BUY, SELL, SPLIT, DIVIDEND
    shares = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    fees = Column(Float, default=0.0)
    notes = Column(String, nullable=True)

    portfolio = relationship("Portfolio", back_populates="trades")

class HoldingsSnapshot(Base):
    __tablename__ = "holdings_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True)
    as_of_date = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=True)

    portfolio = relationship("Portfolio", back_populates="snapshots")

class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    ticker = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="watchlist_items")
    __table_args__ = (UniqueConstraint('user_id', 'ticker', name='uq_watchlist_user_ticker'),)

class AppSettings(Base):
    __tablename__ = "app_settings"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    key = Column(String, primary_key=True)
    value = Column(Text) # JSON serialized

    user = relationship("User", back_populates="settings")

class Run(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True) # UUID
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True)
    status = Column(String, default="running", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    input_hash = Column(String, nullable=True)
    config_hash = Column(String, nullable=True)
    manifest_json = Column(Text, nullable=True)

    portfolio = relationship("Portfolio", back_populates="runs")
    artifacts = relationship("Artifact", back_populates="run")

class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.id"), index=True)
    type = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="artifacts")


class DataCacheIndex(Base):
    __tablename__ = "data_cache_index"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False, index=True)
    key = Column(String, nullable=False, index=True)
    asof_date = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ttl_seconds = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="fresh")
    coverage_pct = Column(Float, nullable=True)
    storage_path = Column(String, nullable=False)
    error_code = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("source", "key", name="uq_cache_source_key"),)
