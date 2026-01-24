from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from typing import Optional
import logging
import json
import csv
import math
import io
from pathlib import Path
import numpy as np
import uuid
from datetime import datetime, timezone

import os

from storage.repo import Repo, use_supabase
from storage import db
from storage import models
from storage.datamanager import data_manager
from storage.db import session_scope
from storage.models import Portfolio
from src.utils_io import ROOT
from src.definitions import DEFINITIONS_REGISTRY
from src.pipeline import compute_app_state, save_artifacts
from src.portfolio import validate_ledger
from market_data.store import MarketDataStore
from market_data.yahoo import fetch_prices
from market_data.fred import get_cached_series
from market_data.persistent_cache import get_or_refresh_frame
from market_data.contracts import MarketDataError
import pandas as pd

logger = logging.getLogger("nexus.api")

app = FastAPI(title="Nexus Analytics API")

def parse_allowed_origins(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return ["http://localhost:3000"]
    origins: list[str] = []
    for origin in raw.split(","):
        cleaned = origin.strip()
        if not cleaned:
            continue
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            continue
        if cleaned.startswith("http://") and "localhost" not in cleaned and "127.0.0.1" not in cleaned:
            continue
        origins.append(cleaned)
    return origins

origins = parse_allowed_origins(os.getenv("NEXUS_ALLOWED_ORIGINS"))
logger.info("CORS allowed origins: %s", ", ".join(origins) if origins else "(none)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def init_db_tables() -> None:
    try:
        engine = db.init_db()
        models.Base.metadata.create_all(bind=engine)
        logger.info("Database initialized (tables ensured).")
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)

repo = Repo()
EXPORTS_DIR_ENV = os.getenv("NEXUS_EXPORT_DIR", "./data/exports")
EXPORTS_DIR = Path(EXPORTS_DIR_ENV)
if not EXPORTS_DIR.is_absolute():
    EXPORTS_DIR = (ROOT / EXPORTS_DIR).resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _validate_run_id(run_id: str) -> None:
    if not run_id or not all(c.isalnum() or c == "-" for c in run_id):
        raise HTTPException(status_code=400, detail="Invalid Run ID")

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load artifact: {exc}")

def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key == "date":
                    cleaned[key] = value
                else:
                    cleaned[key] = _safe_float(value)
            rows.append(cleaned)
    return rows

def _load_manifest(run_id: str) -> dict:
    manifest_path = ROOT / "data" / "cache" / "manifests" / f"{run_id}.json"
    if manifest_path.exists():
        return _load_json(manifest_path)
    run = repo.get_run_by_id(run_id)
    if run and run.manifest_json:
        try:
            return json.loads(run.manifest_json)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Manifest not found")

def _load_summary(run_id: str) -> dict:
    summary_path = EXPORTS_DIR / run_id / f"summary_{run_id}.json"
    if summary_path.exists():
        return _load_json(summary_path)
    summary_path = EXPORTS_DIR / run_id / "summary.json"
    if summary_path.exists():
        return _load_json(summary_path)
    if use_supabase():
        return _load_supabase_json(run_id, "summary.json")
    return _load_json(summary_path)

def _load_performance(run_id: str) -> list[dict]:
    perf_path = EXPORTS_DIR / run_id / "performance.csv"
    if perf_path.exists():
        return _load_csv(perf_path)
    if use_supabase():
        return _load_supabase_csv(run_id, "performance.csv")
    return _load_csv(perf_path)

def _load_monthly_returns(run_id: str) -> list[dict]:
    returns_path = EXPORTS_DIR / run_id / "monthly_returns.csv"
    if returns_path.exists():
        return _load_csv(returns_path)
    if use_supabase():
        return _load_supabase_csv(run_id, "monthly_returns.csv")
    return _load_csv(returns_path)

def _load_attribution_summary(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "attribution_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "attribution_summary.json")
    raise HTTPException(status_code=404, detail="Attribution summary not found")


def _load_attribution_timeseries(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "attribution_timeseries.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "attribution_timeseries.csv")
    return []


def _load_risk_contribution(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "risk_contribution.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "risk_contribution.json")
    raise HTTPException(status_code=404, detail="Risk contribution not found")


def _load_macro_regimes(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "macro_regime_flags.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "macro_regime_flags.csv")
    return []


def _load_macro_summary(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "macro_regime_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "macro_regime_summary.json")
    return {}


def _load_macro_context(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "macro_context.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "macro_context.json")
    return {}


def _load_coverage_summary(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "coverage_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "coverage_summary.json")
    return {}

def _load_risk_free_series(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "risk_free_series.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "risk_free_series.csv")
    return []

def _load_corporate_actions(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "corporate_actions_events.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "corporate_actions_events.csv")
    return []

def _load_data_contracts(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "data_contracts.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "data_contracts.json")
    return {}


def _load_rolling_metrics(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "rolling_metrics.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "rolling_metrics.csv")
    return []


def _load_benchmark_comparison(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "benchmark_comparison.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "benchmark_comparison.json")
    raise HTTPException(status_code=404, detail="Benchmark comparison not found")


def _load_benchmark_timeseries(run_id: str) -> list[dict]:
    path = EXPORTS_DIR / run_id / "benchmark_timeseries.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "benchmark_timeseries.csv")
    return []


def _load_diagnostics(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "diagnostics.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "diagnostics.json")
    return {"diagnostics": [], "run_id": run_id}


def _load_correlation_matrix(run_id: str) -> dict:
    path = EXPORTS_DIR / run_id / "correlation_matrix.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "correlation_matrix.json")
    return {"status": "unavailable", "matrix": {}}

def _load_supabase_json(run_id: str, filename: str) -> dict:
    try:
        data, _content_type = repo.get_artifact_bytes(run_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load artifact: {exc}")

def _load_supabase_csv(run_id: str, filename: str) -> list[dict]:
    try:
        data, _content_type = repo.get_artifact_bytes(run_id, filename)
    except FileNotFoundError:
        return []
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for row in reader:
        cleaned = {}
        for key, value in row.items():
            if key == "date":
                cleaned[key] = value
            else:
                cleaned[key] = _safe_float(value)
        rows.append(cleaned)
    return rows

def _compute_risk_metrics(performance_rows: list[dict], risk_free_rows: list[dict] | None = None) -> dict:
    returns = [row.get("daily_return") for row in performance_rows if row.get("daily_return") is not None]
    if not returns:
        return {"var_95": None, "cvar_95": None, "volatility": None, "sharpe": None}
    arr = np.array(returns, dtype=float)
    var_95 = float(np.quantile(arr, 0.05))
    tail = arr[arr <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else None
    if arr.size < 2:
        return {"var_95": var_95, "cvar_95": cvar_95, "volatility": None, "sharpe": None}
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return {"var_95": var_95, "cvar_95": cvar_95, "volatility": None, "sharpe": None}
    mean = float(np.mean(arr))
    if risk_free_rows:
        perf_df = pd.DataFrame(performance_rows)
        rf_df = pd.DataFrame(risk_free_rows)
        if "date" in perf_df.columns and "date" in rf_df.columns:
            perf_df["date"] = pd.to_datetime(perf_df["date"], errors="coerce")
            rf_df["date"] = pd.to_datetime(rf_df["date"], errors="coerce")
            perf_df = perf_df.dropna(subset=["date", "daily_return"])
            rf_df = rf_df.dropna(subset=["date", "rf_daily_return"])
            aligned = perf_df.set_index("date")[["daily_return"]].join(
                rf_df.set_index("date")[["rf_daily_return"]],
                how="inner",
            )
            if not aligned.empty:
                excess = aligned["daily_return"] - aligned["rf_daily_return"]
                mean = float(excess.mean())
                std = float(excess.std(ddof=1)) if excess.size > 1 else std
    annualized = math.sqrt(252)
    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "volatility": std * annualized,
        "sharpe": (mean / std) * annualized,
    }

def _empty_summary(message: str) -> dict:
    return {
        "source": "",
        "twr": None,
        "mwr": None,
        "final_value": None,
        "last_date": None,
        "max_drawdown": None,
        "errors": [message],
    }

def _load_portfolio(portfolio_id: int) -> dict:
    if use_supabase():
        from storage_supabase.db import session_scope as supa_session_scope
        from storage_supabase import models as supa_models

        with supa_session_scope() as session:
            try:
                portfolio = session.query(supa_models.Portfolio).filter_by(id=portfolio_id).first()
            except Exception:
                raise HTTPException(status_code=404, detail="Portfolio data unavailable (database uninitialized).")
            if not portfolio:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            return {
                "id": portfolio.id,
                "name": portfolio.name,
                "currency": getattr(portfolio, "base_currency", None),
            }

    with session_scope() as session:
        try:
            portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
        except Exception:
            raise HTTPException(status_code=404, detail="Portfolio data unavailable (database uninitialized).")
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "currency": getattr(portfolio, "currency", None) or getattr(portfolio, "base_currency", None),
        }

def _load_ui_state() -> dict:
    path = ROOT / "data" / "user_uploads" / "ui_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _resolve_portfolio_id(portfolio_id: str) -> int:
    if portfolio_id in ("default", "main", "primary"):
        user_id = data_manager.get_current_user_id()
        return data_manager.get_main_portfolio_id(user_id)
    try:
        return int(portfolio_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid portfolio id")

def _load_holdings(portfolio_id: int) -> list[dict]:
    snapshot = data_manager.load_snapshot(portfolio_id)
    holdings: list[dict] = []
    if snapshot.empty:
        watchlist = data_manager.load_watchlist() or []
        return [{"ticker": t} for t in watchlist]
    snapshot = snapshot.rename(columns={"quantity": "shares"}).copy()
    for _, row in snapshot.iterrows():
        holdings.append({
            "ticker": row.get("ticker"),
            "shares": _safe_float(row.get("shares")),
            "cost_basis": _safe_float(row.get("cost_basis")),
        })
    return holdings

def _attach_prices(holdings: list[dict]) -> list[dict]:
    if not holdings:
        return holdings
    enriched: list[dict] = []
    total_value = 0.0
    for holding in holdings:
        ticker = holding.get("ticker")
        price = None
        if ticker:
            ticker_path = ROOT / "reports" / "data" / "ticker" / f"{ticker}.json"
            if ticker_path.exists():
                try:
                    data = json.loads(ticker_path.read_text())
                    price_series = data.get("price", [])
                    if price_series:
                        price = _safe_float(price_series[-1].get("value"))
                except Exception:
                    price = None
        shares = holding.get("shares")
        value = None
        if price is not None and shares is not None:
            value = price * shares
            total_value += value
        enriched.append({**holding, "price": price, "value": value})
    if total_value > 0:
        for holding in enriched:
            value = holding.get("value")
            holding["weight"] = (value / total_value) if value is not None else None
    else:
        for holding in enriched:
            holding["weight"] = None
    return enriched

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Nexus Analytics API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
async def create_run(
    run_type: str = Form("uploaded"),
    portfolio_id: str = Form("default"),
    file: UploadFile | None = File(None),
):
    """
    Create a new portfolio run from an uploaded trades CSV or demo data.
    """
    resolved_portfolio_id = _resolve_portfolio_id(portfolio_id)
    run_id = str(uuid.uuid4())
    run_type_clean = (run_type or "").strip().lower()

    if run_type_clean == "demo":
        try:
            app_state = compute_app_state(
                portfolio_id=resolved_portfolio_id,
                run_id=run_id,
                save_run=True,
                source_override="Demo",
                uploads_active=False,
                run_type="demo",
            )
            save_artifacts(app_state)
            manifest = app_state.run_manifest
            return {
                "run_id": manifest.run_id if manifest else "",
                "status": "completed",
                "timestamp": manifest.timestamp if manifest else None,
            }
        except Exception as exc:
            repo.update_run_failed(run_id, "RUN_COMPUTE_FAILED", str(exc))
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "RUN_COMPUTE_FAILED",
                    "message": "Demo run failed.",
                    "details": {"error": str(exc)},
                    "hint": "Retry later or check market data availability.",
                },
            )

    if not file:
        raise HTTPException(status_code=400, detail="Trades CSV file is required.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
        df = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "shares" in df.columns and "quantity" not in df.columns:
        df = df.rename(columns={"shares": "quantity"})

    validated, errors = validate_ledger(df)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "LEDGER_VALIDATION_FAILED",
                "message": "CSV validation failed.",
                "details": {"errors": errors},
                "hint": "Fix the highlighted CSV columns or values and retry.",
            },
        )

    if "quantity" in validated.columns:
        validated["quantity"] = validated["quantity"].fillna(0)

    if "amount" not in validated.columns:
        def _calc_amount(row: pd.Series) -> float:
            qty = row.get("quantity")
            price = row.get("price")
            if pd.notna(qty) and pd.notna(price):
                return float(qty) * float(price)
            if pd.notna(price):
                return float(price)
            return 0.0
        validated["amount"] = validated.apply(_calc_amount, axis=1)

    data_manager.save_portfolio_inputs(resolved_portfolio_id, validated, None)

    tickers = sorted({t for t in validated["ticker"].astype(str).str.upper().tolist() if t != "CASH"})
    if tickers:
        store = MarketDataStore.default()
        trade_dates = pd.to_datetime(validated["date"], errors="coerce").dt.date.dropna().unique().tolist()
        for ticker in tickers:
            try:
                prices = store.get_prices(
                    ticker,
                    start=str(min(trade_dates)),
                    end=str(max(trade_dates)),
                )
                store.ensure_coverage(prices, trade_dates, ticker)
            except MarketDataError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": exc.error_code,
                        "message": exc.message,
                        "details": exc.details or {},
                        "hint": exc.hint or "Check market data coverage for this ticker.",
                    },
                )

    try:
        app_state = compute_app_state(
            portfolio_id=resolved_portfolio_id,
            run_id=run_id,
            save_run=True,
            source_override="Ledger",
            uploads_active=True,
            run_type="uploaded",
        )
        save_artifacts(app_state)
        manifest = app_state.run_manifest
        return {
            "run_id": manifest.run_id if manifest else "",
            "status": "completed",
            "timestamp": manifest.timestamp if manifest else None,
        }
    except Exception as exc:
        repo.update_run_failed(run_id, "RUN_COMPUTE_FAILED", str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "RUN_COMPUTE_FAILED",
                "message": "Run compute failed.",
                "details": {"error": str(exc)},
                "hint": "Retry later or check market data availability.",
            },
        )

@app.post("/api/v1/run")
async def create_run_alias(
    run_type: str = Form("uploaded"),
    portfolio_id: str = Form("default"),
    file: UploadFile | None = File(None),
):
    return await create_run(run_type=run_type, portfolio_id=portfolio_id, file=file)

@app.get("/runs")
def list_runs():
    runs = [run for run in repo.list_runs() if run.status == "completed"]
    return {
        "runs": [
            {
                "run_id": run.id,
                "status": run.status,
                "timestamp": run.completed_at.isoformat() if run.completed_at else None,
                "input_hash": getattr(run, "input_hash", None),
                "data_hash": getattr(run, "data_hash", None),
            }
            for run in runs
        ]
    }

@app.get("/api/v1/runs")
def list_runs_alias():
    return list_runs()

@app.get("/api/v1/run/latest")
def get_latest_run():
    """
    Get metadata for the last completed run.
    """
    try:
        last_run = repo.get_latest_run()
    except Exception:
        raise HTTPException(status_code=404, detail="No runs found (database empty or uninitialized).")
    if not last_run:
        raise HTTPException(status_code=404, detail="No runs found")
    
    if last_run.manifest_json:
        try:
            payload = json.loads(last_run.manifest_json)
            payload["status"] = last_run.status
            return payload
        except Exception:
            pass
    return {
        "run_id": last_run.run_id,
        "status": last_run.status,
        "timestamp": last_run.completed_at.isoformat() if last_run.completed_at else None,
        "input_hash": last_run.input_hash,
        "data_hash": last_run.data_hash,
    }

@app.get("/latest-run")
def latest_run_alias():
    return get_latest_run()

@app.get("/api/v1/run/{run_id}/summary")
def get_run_summary(run_id: str):
    """
    Serve the precomputed JSON summary artifact.
    """
    # Security: Validate run_id format to prevent traversal (basic)
    _validate_run_id(run_id)
    return _load_summary(run_id)

@app.get("/api/v1/run/{run_id}")
def get_run(run_id: str):
    _validate_run_id(run_id)
    manifest = _load_manifest(run_id)
    try:
        summary = _load_summary(run_id)
    except HTTPException:
        summary = _empty_summary("Summary artifact not available.")
    try:
        performance = _load_performance(run_id)
    except HTTPException:
        performance = []
    try:
        monthly_returns = _load_monthly_returns(run_id)
    except HTTPException:
        monthly_returns = []
    try:
        attribution_summary = _load_attribution_summary(run_id)
    except HTTPException:
        attribution_summary = {}
    attribution_timeseries = _load_attribution_timeseries(run_id)
    try:
        risk_contribution = _load_risk_contribution(run_id)
    except HTTPException:
        risk_contribution = {"summary": {}, "contributions": []}
    rolling_metrics = _load_rolling_metrics(run_id)
    macro_regimes = _load_macro_regimes(run_id)
    macro_summary = _load_macro_summary(run_id)
    macro_context = _load_macro_context(run_id)
    try:
        benchmark_comparison = _load_benchmark_comparison(run_id)
    except HTTPException:
        benchmark_comparison = {}
    benchmark_timeseries = _load_benchmark_timeseries(run_id)
    diagnostics_payload = _load_diagnostics(run_id)
    correlation_matrix = _load_correlation_matrix(run_id)
    coverage_summary = _load_coverage_summary(run_id)
    risk_free_series = _load_risk_free_series(run_id)
    corporate_actions = _load_corporate_actions(run_id)
    data_contracts = _load_data_contracts(run_id)
    risk = _compute_risk_metrics(performance, risk_free_series)
    equity_curve = [
        {"date": row.get("date"), "value": row.get("value")}
        for row in performance
        if row.get("date") is not None and row.get("value") is not None
    ]
    return {
        "manifest": manifest,
        "summary": summary,
        "coverage_summary": coverage_summary,
        "risk_free_series": risk_free_series,
        "corporate_actions": corporate_actions,
        "data_contracts": data_contracts,
        "equity_curve": equity_curve,
        "performance": performance,
        "monthly_returns": monthly_returns,
        "risk": risk,
        "attribution_summary": attribution_summary,
        "attribution_timeseries": attribution_timeseries,
        "risk_contribution": risk_contribution,
        "rolling_metrics": rolling_metrics,
        "macro_regimes": macro_regimes,
        "macro": {
            "status": (
                "sufficient"
                if (macro_context.get("status") or macro_summary.get("status")) == "ok"
                else macro_context.get("status") or macro_summary.get("status", "unavailable")
            ),
            "available_series": macro_context.get("available_series", []),
            "missing_series": macro_context.get("missing_series", macro_summary.get("missing_series", [])),
            "tags": macro_context.get("tags", []),
            "warnings": macro_context.get("warnings", []),
            "as_of": macro_context.get("as_of", macro_summary.get("as_of")),
            "cache_status": macro_context.get("cache_status", {}),
            "flags": macro_regimes,
        },
        "benchmark_comparison": benchmark_comparison,
        "benchmark_timeseries": benchmark_timeseries,
        "diagnostics": diagnostics_payload.get("diagnostics", []),
        "correlation_matrix": correlation_matrix,
    }

@app.get("/run/{run_id}")
def run_alias(run_id: str):
    return get_run(run_id)

@app.get("/api/v1/portfolio/{portfolio_id}")
def get_portfolio(portfolio_id: str):
    resolved_id = _resolve_portfolio_id(portfolio_id)
    portfolio = _load_portfolio(resolved_id)
    holdings = _attach_prices(_load_holdings(resolved_id))
    ui_state = _load_ui_state()
    portfolio["benchmark"] = ui_state.get("benchmark")
    return {"portfolio": portfolio, "holdings": holdings}

@app.get("/portfolio/{portfolio_id}")
def portfolio_alias(portfolio_id: str):
    return get_portfolio(portfolio_id)

@app.get("/api/v1/definitions")
def get_definitions():
    return DEFINITIONS_REGISTRY

@app.get("/definitions")
def definitions_alias():
    return get_definitions()


def _store_warmup_report(report: dict) -> None:
    timestamp = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    filename = f"warmup_{timestamp.replace(':', '').replace('-', '')}.json"
    local_dir = ROOT / "data" / "cache" / "warmup_reports"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename
    local_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if use_supabase():
        try:
            from storage_supabase.storage import upload_bytes

            bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")
            storage_path = f"system/warmup/{filename}"
            upload_bytes(bucket, storage_path, local_path.read_bytes(), "application/json")
        except Exception:
            logger.exception("Failed to upload warmup report to Supabase storage.")


@app.post("/admin/warmup")
def warmup(payload: dict | None = None, x_admin_key: str | None = Header(default=None)):
    admin_key = os.getenv("ADMIN_WARMUP_KEY")
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Unauthorized")
    payload = payload or {}
    benchmarks = payload.get("benchmarks") or ["SPY"]
    force_refresh = bool(payload.get("force"))

    series_ids = ["CPIAUCSL", "DFF", "VIXCLS"]
    series_status: dict[str, dict] = {}
    for series_id in series_ids:
        result = get_cached_series(series_id, allow_refresh=True, force_refresh=force_refresh)
        series_status[series_id] = {
            "status": result.status,
            "rows": int(result.frame.shape[0]) if result.frame is not None else 0,
        }

    bench_status: dict[str, dict] = {}
    for ticker in benchmarks:
        try:
            result = get_or_refresh_frame(
                source="yahoo",
                key=ticker,
                ttl_seconds=21600,
                fetch_fn=lambda t=ticker: fetch_prices(
                    t, start="2015-01-01", end=datetime.now(timezone.utc).strftime("%Y-%m-%d")
                ),
                asof_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                allow_refresh=True,
                force_refresh=force_refresh,
            )
            bench_status[ticker] = {
                "status": result.status,
                "rows": int(result.frame.shape[0]) if result.frame is not None else 0,
            }
        except Exception as exc:
            bench_status[ticker] = {"status": "error", "error": str(exc)}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "force_refresh": force_refresh,
        "series": series_status,
        "benchmarks": bench_status,
    }
    _store_warmup_report(report)
    return report
@app.get("/api/v1/run/{run_id}/export/{artifact}")
def export_artifact(run_id: str, artifact: str):
    _validate_run_id(run_id)
    allowed = {
        "summary": "summary.json",
        "performance": "performance.csv",
        "monthly-returns": "monthly_returns.csv",
        "report": "report.html",
        "summary-json": "summary.json",
        "coverage-summary": "coverage_summary.json",
        "risk-free-series": "risk_free_series.csv",
        "corporate-actions": "corporate_actions_events.csv",
        "data-contracts": "data_contracts.json",
        "performance-csv": "performance.csv",
        "monthly-returns-csv": "monthly_returns.csv",
        "attribution-summary": "attribution_summary.json",
        "attribution-timeseries": "attribution_timeseries.csv",
        "risk-contribution": "risk_contribution.csv",
        "risk-contribution-json": "risk_contribution.json",
        "macro-regimes": "macro_regime_flags.csv",
        "macro-regime-summary": "macro_regime_summary.json",
        "macro-context": "macro_context.json",
        "rolling-metrics": "rolling_metrics.csv",
        "benchmark-comparison": "benchmark_comparison.json",
        "benchmark-timeseries": "benchmark_timeseries.csv",
        "diagnostics": "diagnostics.json",
        "correlation-matrix": "correlation_matrix.json",
    }
    filename = allowed.get(artifact)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    if use_supabase():
        try:
            data, content_type = repo.get_artifact_bytes(run_id, filename)
            return Response(content=data, media_type=content_type)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Artifact not found. Run compute to generate exports.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load artifact: {exc}")

    artifact_path = EXPORTS_DIR / run_id / filename
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found. Run compute to generate exports.")
    return FileResponse(artifact_path)

@app.get("/run/{run_id}/export/{artifact}")
def export_artifact_alias(run_id: str, artifact: str):
    return export_artifact(run_id, artifact)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
