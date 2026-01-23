from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, delete

from storage.models import User, Portfolio, Trade, HoldingsSnapshot, WatchlistItem, Run, Artifact, AppSettings
from storage.db import session_scope

def get_or_create_default_user() -> User:
    with session_scope() as session:
        user = session.query(User).filter_by(username="default").first()
        if not user:
            user = User(username="default")
            session.add(user)
            session.commit()
            session.refresh(user)
        session.expunge(user)
        return user

def get_user_id() -> int:
    # Helper ensuring we have the ID for 'default' user
    with session_scope() as session:
         user = session.query(User).filter_by(username="default").first()
         if not user:
             user = User(username="default")
             session.add(user)
             session.commit()
         return user.id

def get_default_portfolio_id(user_id: int) -> int:
    with session_scope() as session:
        port = session.query(Portfolio).filter_by(user_id=user_id, name="Main Portfolio").first()
        if not port:
            port = Portfolio(user_id=user_id, name="Main Portfolio")
            session.add(port)
            session.commit()
        return port.id

def list_watch_tickers(user_id: int) -> list[str]:
    with session_scope() as session:
        items = session.query(WatchlistItem).filter_by(user_id=user_id).all()
        return [item.ticker for item in items]

def replace_watchlist(user_id: int, tickers: list[str]):
    with session_scope() as session:
        # Delete existing
        session.query(WatchlistItem).filter_by(user_id=user_id).delete()
        # Add new
        for t in tickers:
            session.add(WatchlistItem(user_id=user_id, ticker=t))

def replace_trades(portfolio_id: int, trades_dicts: list[dict]):
    """
    trades_dicts expected to match Trade model fields: date, ticker, action, shares, amount, etc.
    """
    with session_scope() as session:
        session.query(Trade).filter_by(portfolio_id=portfolio_id).delete()
        for t_data in trades_dicts:
            trade = Trade(portfolio_id=portfolio_id, **t_data)
            session.add(trade)

def replace_snapshot(portfolio_id: int, snapshot_dicts: list[dict]):
    with session_scope() as session:
        session.query(HoldingsSnapshot).filter_by(portfolio_id=portfolio_id).delete()
        for s_data in snapshot_dicts:
            snap = HoldingsSnapshot(portfolio_id=portfolio_id, **s_data)
            session.add(snap)

def get_trades(portfolio_id: int) -> list[dict]:
    with session_scope() as session:
        trades = session.query(Trade).filter_by(portfolio_id=portfolio_id).all()
        # Return dicts to avoid detached instance errors
        return [
            {
                "date": t.date,
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "amount": t.amount,
                "price": t.price,
                "fees": t.fees,
                "notes": t.notes
            }
            for t in trades
        ]

def get_latest_snapshot(portfolio_id: int) -> list[dict]:
    with session_scope() as session:
        snaps = session.query(HoldingsSnapshot).filter_by(portfolio_id=portfolio_id).all()
        return [
            {
                "as_of_date": s.as_of_date,
                "ticker": s.ticker,
                "shares": s.shares,
                "cost_basis": s.cost_basis
            }
            for s in snaps
        ]

def create_run(run_id: str, portfolio_id: int, input_hash: str | None, config_hash: str | None):
    with session_scope() as session:
        run = Run(
            id=run_id,
            portfolio_id=portfolio_id,
            status="running",
            created_at=datetime.utcnow(),
            input_hash=input_hash,
            config_hash=config_hash
        )
        session.add(run)

def update_run_complete(run_id: str, manifest_json: str):
    with session_scope() as session:
        run = session.query(Run).filter_by(id=run_id).first()
        if run:
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.manifest_json = manifest_json

def add_artifact(run_id: str, artifact_type: str, path: str):
    with session_scope() as session:
        artifact = Artifact(run_id=run_id, type=artifact_type, path=path)
        session.add(artifact)


# Repo class wrapper for FastAPI compatibility
class Repo:
    """
    Repository class wrapper for FastAPI compatibility.
    Wraps existing function-based API.
    """
    
    def get_latest_run(self):
        """Get the most recent completed run."""
        with session_scope() as session:
            run = session.query(Run).filter_by(status="completed").order_by(Run.completed_at.desc()).first()
            if run:
                session.expunge(run)
            return run
    
    def get_run_by_id(self, run_id: str):
        """Get a specific run by ID."""
        with session_scope() as session:
            run = session.query(Run).filter_by(id=run_id).first()
            if run:
                session.expunge(run)
            return run

    def list_runs(self, limit: int = 50):
        """List recent runs, newest first."""
        with session_scope() as session:
            runs = (
                session.query(Run)
                .order_by(Run.completed_at.desc().nullslast(), Run.created_at.desc())
                .limit(limit)
                .all()
            )
            for run in runs:
                session.expunge(run)
            return runs
