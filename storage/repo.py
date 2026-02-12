from __future__ import annotations

import os
from datetime import datetime

from contextlib import contextmanager
from storage.models import User, Portfolio, Trade, HoldingsSnapshot, WatchlistItem, Run, Artifact
from storage.db import session_scope


@contextmanager
def _use_session(external_session=None):
    """Use the provided session or create a new one via session_scope."""
    if external_session is not None:
        yield external_session
    else:
        with session_scope() as s:
            yield s


def _supabase_enabled() -> bool:
    return bool(
        os.getenv("SUPABASE_DB_URL")
        and os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )


class LocalRepoBackend:
    def get_or_create_default_user(self) -> User:
        with session_scope() as session:
            user = session.query(User).filter_by(username="default").first()
            if not user:
                user = User(username="default")
                session.add(user)
                session.commit()
                session.refresh(user)
            session.expunge(user)
            return user

    def get_user_id(self) -> int:
        with session_scope() as session:
            user = session.query(User).filter_by(username="default").first()
            if not user:
                user = User(username="default")
                session.add(user)
                session.commit()
            return user.id

    def get_default_portfolio_id(self, user_id: int) -> int:
        with session_scope() as session:
            port = session.query(Portfolio).filter_by(user_id=user_id, name="Main Portfolio").first()
            if not port:
                port = Portfolio(user_id=user_id, name="Main Portfolio")
                session.add(port)
                session.commit()
            return port.id

    def list_watch_tickers(self, user_id: int) -> list[str]:
        with session_scope() as session:
            items = session.query(WatchlistItem).filter_by(user_id=user_id).all()
            return [item.ticker for item in items]

    def replace_watchlist(self, user_id: int, tickers: list[str]):
        with session_scope() as session:
            session.query(WatchlistItem).filter_by(user_id=user_id).delete()
            for t in tickers:
                session.add(WatchlistItem(user_id=user_id, ticker=t))

    def replace_trades(self, portfolio_id: int, trades_dicts: list[dict]):
        with session_scope() as session:
            session.query(Trade).filter_by(portfolio_id=portfolio_id).delete()
            for t_data in trades_dicts:
                trade = Trade(portfolio_id=portfolio_id, **t_data)
                session.add(trade)

    def replace_snapshot(self, portfolio_id: int, snapshot_dicts: list[dict]):
        with session_scope() as session:
            session.query(HoldingsSnapshot).filter_by(portfolio_id=portfolio_id).delete()
            for s_data in snapshot_dicts:
                snap = HoldingsSnapshot(portfolio_id=portfolio_id, **s_data)
                session.add(snap)

    def get_trades(self, portfolio_id: int) -> list[dict]:
        with session_scope() as session:
            trades = session.query(Trade).filter_by(portfolio_id=portfolio_id).all()
            return [
                {
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares,
                    "amount": t.amount,
                    "price": t.price,
                    "fees": t.fees,
                    "notes": t.notes,
                }
                for t in trades
            ]

    def get_latest_snapshot(self, portfolio_id: int) -> list[dict]:
        with session_scope() as session:
            snaps = session.query(HoldingsSnapshot).filter_by(portfolio_id=portfolio_id).all()
            return [
                {
                    "as_of_date": s.as_of_date,
                    "ticker": s.ticker,
                    "shares": s.shares,
                    "cost_basis": s.cost_basis,
                }
                for s in snaps
            ]

    def create_run(self, run_id: str, portfolio_id: int, input_hash: str | None, config_hash: str | None, run_type: str | None = None, session=None):
        with _use_session(session) as s:
            run = Run(
                id=run_id,
                portfolio_id=portfolio_id,
                status="running",
                created_at=datetime.utcnow(),
                input_hash=input_hash,
                config_hash=config_hash,
            )
            s.add(run)
            if session is not None:
                s.flush()

    def update_run_complete(self, run_id: str, manifest_json: str, run_type: str | None = None, session=None):
        with _use_session(session) as s:
            run = s.query(Run).filter_by(id=run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                run.manifest_json = manifest_json
            if session is not None:
                s.flush()

    def update_run_failed(self, run_id: str, error_code: str | None = None, message: str | None = None, session=None):
        with _use_session(session) as s:
            run = s.query(Run).filter_by(id=run_id).first()
            if run:
                run.status = "failed"
                run.completed_at = datetime.utcnow()
            if session is not None:
                s.flush()

    def add_artifact(self, run_id: str, artifact_type: str, path: str, session=None):
        with _use_session(session) as s:
            artifact = Artifact(run_id=run_id, type=artifact_type, path=path)
            s.add(artifact)
            if session is not None:
                s.flush()

    def get_latest_run(self):
        with session_scope() as session:
            run = session.query(Run).filter_by(status="completed").order_by(Run.completed_at.desc()).first()
            if run:
                session.expunge(run)
            return run

    def get_run_by_id(self, run_id: str):
        with session_scope() as session:
            run = session.query(Run).filter_by(id=run_id).first()
            if run:
                session.expunge(run)
            return run

    def list_runs(self, limit: int = 50):
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

    def get_artifact_bytes(self, run_id: str, filename: str):
        raise FileNotFoundError("Artifact bytes not available in local repo.")


def _select_backend():
    if _supabase_enabled():
        from storage_supabase.repo import SupabaseRepo

        return SupabaseRepo()
    return LocalRepoBackend()


_backend = _select_backend()


def use_supabase() -> bool:
    return _supabase_enabled()


def get_or_create_default_user() -> User:
    return _backend.get_or_create_default_user()


def get_user_id() -> int:
    return _backend.get_user_id()


def get_default_portfolio_id(user_id: int) -> int:
    return _backend.get_default_portfolio_id(user_id)


def list_watch_tickers(user_id: int) -> list[str]:
    return _backend.list_watch_tickers(user_id)


def replace_watchlist(user_id: int, tickers: list[str]):
    return _backend.replace_watchlist(user_id, tickers)


def replace_trades(portfolio_id: int, trades_dicts: list[dict]):
    return _backend.replace_trades(portfolio_id, trades_dicts)


def replace_snapshot(portfolio_id: int, snapshot_dicts: list[dict]):
    return _backend.replace_snapshot(portfolio_id, snapshot_dicts)


def get_trades(portfolio_id: int) -> list[dict]:
    return _backend.get_trades(portfolio_id)


def get_latest_snapshot(portfolio_id: int) -> list[dict]:
    return _backend.get_latest_snapshot(portfolio_id)


def create_run(run_id: str, portfolio_id: int, input_hash: str | None, config_hash: str | None, run_type: str | None = None, session=None):
    return _backend.create_run(run_id, portfolio_id, input_hash, config_hash, run_type, session=session)


def update_run_complete(run_id: str, manifest_json: str, run_type: str | None = None, session=None):
    return _backend.update_run_complete(run_id, manifest_json, run_type, session=session)


def update_run_failed(run_id: str, error_code: str | None = None, message: str | None = None, session=None):
    return _backend.update_run_failed(run_id, error_code, message, session=session)


def add_artifact(run_id: str, artifact_type: str, path: str, session=None):
    return _backend.add_artifact(run_id, artifact_type, path, session=session)


class Repo:
    def get_latest_run(self):
        return _backend.get_latest_run()

    def get_run_by_id(self, run_id: str):
        return _backend.get_run_by_id(run_id)

    def list_runs(self, limit: int = 50):
        return _backend.list_runs(limit)

    def get_artifact_bytes(self, run_id: str, filename: str):
        return _backend.get_artifact_bytes(run_id, filename)

    def update_run_failed(self, run_id: str, error_code: str | None = None, message: str | None = None):
        return _backend.update_run_failed(run_id, error_code, message)
