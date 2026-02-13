from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import json
import logging

import pandas as pd

from storage_supabase.db import session_scope
from storage_supabase import models
from storage_supabase.storage import upload_bytes, download_bytes


class SupabaseRepo:
    def __init__(self) -> None:
        self.bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")

    def _service_context_user_id(self) -> str:
        explicit = os.getenv("SUPABASE_SERVICE_CONTEXT_USER_ID")
        if explicit:
            return explicit
        legacy = os.getenv("SUPABASE_DEFAULT_USER_ID")
        if legacy:
            logging.getLogger("nexus.storage_supabase").warning(
                "SUPABASE_DEFAULT_USER_ID is deprecated; use SUPABASE_SERVICE_CONTEXT_USER_ID."
            )
            return legacy
        raise RuntimeError(
            "No service context user id configured. Set SUPABASE_SERVICE_CONTEXT_USER_ID for non-request jobs."
        )

    # Portfolio
    def get_or_create_default_user(self):
        class _UserStub:
            id = ""
            username = "default"

        user = _UserStub()
        user.id = self._service_context_user_id()
        return user

    def get_user_id(self) -> str:
        return self._service_context_user_id()

    def get_default_portfolio_id(self, user_id: str | int) -> int:
        resolved_user_id = str(user_id)
        with session_scope() as session:
            portfolio = (
                session.query(models.Portfolio)
                .filter_by(user_id=resolved_user_id, name="Main Portfolio")
                .order_by(models.Portfolio.id.asc())
                .first()
            )
            if not portfolio:
                portfolio = models.Portfolio(
                    user_id=resolved_user_id,
                    name="Main Portfolio",
                    base_currency="USD",
                )
                session.add(portfolio)
                session.flush()
            return portfolio.id

    def list_watch_tickers(self, _user_id: int) -> list[str]:
        return []

    def replace_watchlist(self, _user_id: int, _tickers: list[str]):
        return None

    # Trades
    def replace_trades(self, portfolio_id: int, trades_dicts: list[dict]):
        with session_scope() as session:
            session.query(models.Transaction).filter_by(portfolio_id=portfolio_id).delete()
            for t_data in trades_dicts:
                trade = models.Transaction(
                    portfolio_id=portfolio_id,
                    date=t_data.get("date"),
                    ticker=t_data.get("ticker"),
                    action=t_data.get("action"),
                    quantity=t_data.get("shares") or 0.0,
                    price=t_data.get("price"),
                    fees=t_data.get("fees"),
                    amount=t_data.get("amount"),
                )
                session.add(trade)

    def get_trades(self, portfolio_id: int) -> list[dict]:
        with session_scope() as session:
            trades = session.query(models.Transaction).filter_by(portfolio_id=portfolio_id).all()
            return [
                {
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.quantity,
                    "amount": t.amount,
                    "price": t.price,
                    "fees": t.fees,
                    "notes": None,
                }
                for t in trades
            ]

    def replace_snapshot(self, portfolio_id: int, snapshot_dicts: list[dict]):
        with session_scope() as session:
            session.query(models.HoldingsSnapshot).filter_by(portfolio_id=portfolio_id).delete()
            for s_data in snapshot_dicts:
                snap = models.HoldingsSnapshot(
                    portfolio_id=portfolio_id,
                    as_of_date=s_data.get("as_of_date"),
                    ticker=s_data.get("ticker"),
                    shares=s_data.get("shares") or 0.0,
                    cost_basis=s_data.get("cost_basis"),
                )
                session.add(snap)

    def get_latest_snapshot(self, portfolio_id: int) -> list[dict]:
        with session_scope() as session:
            snaps = (
                session.query(models.HoldingsSnapshot)
                .filter_by(portfolio_id=portfolio_id)
                .order_by(models.HoldingsSnapshot.as_of_date.desc())
                .all()
            )
            return [
                {
                    "as_of_date": s.as_of_date,
                    "ticker": s.ticker,
                    "shares": s.shares,
                    "cost_basis": s.cost_basis,
                }
                for s in snaps
            ]

    # Runs
    def create_run(
        self,
        run_id: str,
        portfolio_id: int,
        input_hash: str | None,
        config_hash: str | None,
        run_type: str | None = None,
        session=None,
    ):
        with session_scope() as session:
            run = models.Run(
                id=run_id,
                portfolio_id=portfolio_id,
                status="running",
                created_at=datetime.utcnow(),
                run_type=run_type,
            )
            session.add(run)

    def update_run_complete(self, run_id: str, manifest_json: str, run_type: str | None = None, session=None):
        with session_scope() as session:
            run = session.query(models.Run).filter_by(id=run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                run.manifest_json = manifest_json
                if run_type:
                    run.run_type = run_type

    def update_run_failed(self, run_id: str, error_code: str | None = None, message: str | None = None, session=None):
        with session_scope() as session:
            run = session.query(models.Run).filter_by(id=run_id).first()
            if run:
                run.status = "failed"
                run.completed_at = datetime.utcnow()
                run.error_code = error_code
                run.message = message

    def _runs_query(self, session, user_id: str | None = None):
        query = session.query(models.Run)
        if user_id is not None:
            query = query.join(models.Portfolio, models.Run.portfolio_id == models.Portfolio.id).filter(
                models.Portfolio.user_id == str(user_id)
            )
        return query

    def get_latest_run(self, user_id: str | None = None):
        with session_scope() as session:
            run = (
                self._runs_query(session, user_id=user_id)
                .filter(models.Run.status == "completed")
                .order_by(models.Run.completed_at.desc())
                .first()
            )
            if run:
                session.expunge(run)
            return run

    def get_run_by_id(self, run_id: str, user_id: str | None = None):
        with session_scope() as session:
            run = self._runs_query(session, user_id=user_id).filter(models.Run.id == run_id).first()
            if run:
                session.expunge(run)
            return run

    def list_runs(self, limit: int = 50, user_id: str | None = None):
        with session_scope() as session:
            runs = (
                self._runs_query(session, user_id=user_id)
                .order_by(models.Run.completed_at.desc().nullslast(), models.Run.created_at.desc())
                .limit(limit)
                .all()
            )
            for run in runs:
                session.expunge(run)
            return runs

    # Artifacts
    def add_artifact(self, run_id: str, artifact_type: str, path: str, session=None):
        file_path = Path(path)
        if not file_path.exists():
            return
        data = file_path.read_bytes()
        storage_path = f"runs/{run_id}/{file_path.name}"
        content_type = "application/octet-stream"
        if file_path.suffix == ".json":
            content_type = "application/json"
        elif file_path.suffix == ".csv":
            content_type = "text/csv"
        elif file_path.suffix == ".html":
            content_type = "text/html"

        upload_bytes(self.bucket, storage_path, data, content_type)

        with session_scope() as session:
            artifact = models.RunArtifact(
                run_id=run_id,
                artifact_key=file_path.name,
                storage_path=storage_path,
                content_type=content_type,
            )
            session.add(artifact)

    def get_artifact_bytes(self, run_id: str, filename: str, user_id: str | None = None) -> tuple[bytes, str]:
        with session_scope() as session:
            query = (
                session.query(models.RunArtifact)
                .join(models.Run, models.RunArtifact.run_id == models.Run.id)
                .filter(models.RunArtifact.run_id == run_id, models.RunArtifact.artifact_key == filename)
            )
            if user_id is not None:
                query = query.join(models.Portfolio, models.Run.portfolio_id == models.Portfolio.id).filter(
                    models.Portfolio.user_id == str(user_id)
                )
            artifact = query.first()
            if not artifact:
                raise FileNotFoundError("Artifact not found")
            data = download_bytes(self.bucket, artifact.storage_path)
            return data, artifact.content_type or "application/octet-stream"
