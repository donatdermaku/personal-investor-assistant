from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import json

import pandas as pd

from storage_supabase.db import session_scope
from storage_supabase import models
from storage_supabase.storage import upload_bytes, download_bytes


class SupabaseRepo:
    def __init__(self) -> None:
        self.bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")

    # Portfolio
    def get_or_create_default_user(self):
        class _UserStub:
            id = 1
            username = "default"

        return _UserStub()

    def get_user_id(self) -> int:
        return 1

    def get_default_portfolio_id(self, _user_id: int) -> int:
        with session_scope() as session:
            portfolio = session.query(models.Portfolio).order_by(models.Portfolio.id.asc()).first()
            if not portfolio:
                portfolio = models.Portfolio(name="Main Portfolio", base_currency="USD")
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

    def replace_snapshot(self, _portfolio_id: int, _snapshot_dicts: list[dict]):
        return None

    def get_latest_snapshot(self, _portfolio_id: int) -> list[dict]:
        return []

    # Runs
    def create_run(self, run_id: str, portfolio_id: int, input_hash: str | None, config_hash: str | None, run_type: str | None = None):
        with session_scope() as session:
            run = models.Run(
                id=run_id,
                portfolio_id=portfolio_id,
                status="running",
                created_at=datetime.utcnow(),
                run_type=run_type,
            )
            session.add(run)

    def update_run_complete(self, run_id: str, manifest_json: str, run_type: str | None = None):
        with session_scope() as session:
            run = session.query(models.Run).filter_by(id=run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                run.manifest_json = manifest_json
                if run_type:
                    run.run_type = run_type

    def get_latest_run(self):
        with session_scope() as session:
            return session.query(models.Run).filter_by(status="completed").order_by(models.Run.completed_at.desc()).first()

    def get_run_by_id(self, run_id: str):
        with session_scope() as session:
            return session.query(models.Run).filter_by(id=run_id).first()

    def list_runs(self, limit: int = 50):
        with session_scope() as session:
            return (
                session.query(models.Run)
                .order_by(models.Run.completed_at.desc().nullslast(), models.Run.created_at.desc())
                .limit(limit)
                .all()
            )

    # Artifacts
    def add_artifact(self, run_id: str, artifact_type: str, path: str):
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

    def get_artifact_bytes(self, run_id: str, filename: str) -> tuple[bytes, str]:
        with session_scope() as session:
            artifact = (
                session.query(models.RunArtifact)
                .filter_by(run_id=run_id, artifact_key=filename)
                .first()
            )
            if not artifact:
                raise FileNotFoundError("Artifact not found")
            data = download_bytes(self.bucket, artifact.storage_path)
            return data, artifact.content_type or "application/octet-stream"
