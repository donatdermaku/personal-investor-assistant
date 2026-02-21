from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from typing import Optional, Any
import logging
import json
import csv
import math
import io
import base64
import hashlib
import hmac
import gc
from pathlib import Path
from collections import deque
import numpy as np
import uuid
from datetime import datetime, timezone
import time
import threading
import sys
import resource
import requests


import os

from storage.repo import Repo, use_supabase
from storage import db
from storage import models
from storage.datamanager import data_manager
from storage.db import session_scope
from storage.models import Portfolio
from src.utils_io import ROOT
from src.utils_memory import log_rss
from src.analytics.metrics_registry import (
    assert_metric_artifact_aliases_registered,
    assert_run_metric_payload_keys_registered,
    get_exposed_definitions,
)
from src.pipeline import compute_app_state, save_artifacts
from src.portfolio import validate_ledger
from src.streamlit_export import html_to_pdf_bytes
from market_data.store import MarketDataStore
from market_data.yahoo import fetch_prices
from market_data.fred import get_cached_series
from market_data.persistent_cache import get_or_refresh_frame
from market_data.contracts import MarketDataError

# Service layer
from src.services.portfolio_service import PortfolioService
from src.services.market_data_service import MarketDataService

import pandas as pd

logger = logging.getLogger("nexus.api")

app = FastAPI(title="Nexus Analytics API")
APP_STARTED_AT = datetime.now(timezone.utc)


class SlidingWindowRateLimiter:
    """Simple in-memory sliding window rate limiter by client IP."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float) -> tuple[bool, int, int]:
        cutoff = now - self.window_seconds
        with self._lock:
            entries = [ts for ts in self._requests.get(key, []) if ts > cutoff]
            if len(entries) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - entries[0])))
                remaining = 0
                self._requests[key] = entries
                return False, retry_after, remaining
            entries.append(now)
            self._requests[key] = entries
            remaining = max(0, self.limit - len(entries))
            return True, 0, remaining


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


RATE_LIMIT_ENABLED = _read_bool_env("NEXUS_RATE_LIMIT_ENABLED", True)
RATE_LIMIT_PER_WINDOW = int(os.getenv("NEXUS_RATE_LIMIT_PER_WINDOW", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_EXEMPT_PATHS = {"/", "/health", "/ops/health"}
_rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_PER_WINDOW, RATE_LIMIT_WINDOW_SECONDS)
TIMING_LOG_ENABLED = _read_bool_env("NEXUS_TIMING_LOG_ENABLED", True)
DEFINITIONS_CACHE_TTL_SECONDS = int(os.getenv("NEXUS_DEFINITIONS_CACHE_TTL_SECONDS", "300"))
OPS_CACHE_STATS_TTL_SECONDS = int(os.getenv("NEXUS_OPS_CACHE_STATS_TTL_SECONDS", "30"))
ERROR_EVENTS_MAX = int(os.getenv("NEXUS_ERROR_EVENTS_MAX", "200"))


class TtlValueCache:
    """Thread-safe in-memory TTL cache for small, read-heavy payloads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, ttl_seconds: int) -> Any | None:
        if ttl_seconds <= 0:
            return None
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            ts, value = entry
            if now - ts > ttl_seconds:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)


_ttl_cache = TtlValueCache()
_error_events: deque[dict[str, Any]] = deque(maxlen=max(10, ERROR_EVENTS_MAX))


def _decode_segment(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _decode_and_verify_hs256_jwt(token: str, secret: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid bearer token format.")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_decode_segment(header_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid bearer token header.")
    if header.get("alg") != "HS256":
        raise HTTPException(status_code=401, detail="Unsupported bearer token algorithm.")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        provided = _decode_segment(signature_b64)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid bearer token signature encoding.")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid bearer token signature.")

    try:
        payload = json.loads(_decode_segment(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid bearer token payload.")

    exp = payload.get("exp")
    if exp is not None:
        try:
            if int(exp) <= int(time.time()):
                raise HTTPException(status_code=401, detail="Bearer token expired.")
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid bearer token expiration.")

    return payload


def _decode_token_header(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid bearer token format.")
    header_b64 = parts[0]
    try:
        return json.loads(_decode_segment(header_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid bearer token header.")


def _verify_token_with_supabase_auth(token: str) -> dict:
    supabase_url = os.getenv("SUPABASE_URL")
    api_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not api_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase auth verification requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY.",
        )

    user_url = f"{supabase_url.rstrip('/')}/auth/v1/user"
    try:
        resp = requests.get(
            user_url,
            headers={"Authorization": f"Bearer {token}", "apikey": api_key},
            timeout=5,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Supabase auth verification unavailable.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid bearer token.")

    try:
        user_payload = resp.json()
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Supabase auth response.")

    user_id = user_payload.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Bearer token missing subject.")

    return {
        "sub": str(user_id),
        "email": user_payload.get("email"),
        "aud": user_payload.get("aud"),
        "iss": user_payload.get("iss"),
    }


def _decode_and_verify_supabase_jwt(token: str) -> dict:
    header = _decode_token_header(token)
    alg = str(header.get("alg", "")).upper()
    if alg == "HS256":
        jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        if not jwt_secret:
            raise HTTPException(status_code=503, detail="SUPABASE_JWT_SECRET not configured.")
        return _decode_and_verify_hs256_jwt(token, jwt_secret)
    return _verify_token_with_supabase_auth(token)


def get_current_user(request: Request) -> dict:
    """
    Resolve authenticated user from Supabase JWT in Authorization header.
    For non-Supabase mode, returns a local pseudo-user.
    """
    if not use_supabase():
        return {"user_id": "local-dev", "auth_mode": "local"}

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer token.")

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    payload = getattr(request.state, "bearer_payload", None)
    if not payload:
        payload = _decode_and_verify_supabase_jwt(token)
        request.state.bearer_payload = payload
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Bearer token missing subject.")

    return {"user_id": str(user_id), "claims": payload, "auth_mode": "supabase"}

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


def resolve_allowed_origins(raw: str | None, env_name: str | None) -> list[str]:
    env_norm = (env_name or "development").strip().lower()
    if env_norm == "production" and (raw is None or raw.strip() == ""):
        logger.warning(
            "NEXUS_ALLOWED_ORIGINS is empty in production; CORS allowlist defaults to empty."
        )
        return []
    return parse_allowed_origins(raw)


def _admin_key_valid(x_admin_key: str | None) -> bool:
    admin_key = os.getenv("ADMIN_WARMUP_KEY")
    return bool(admin_key and x_admin_key == admin_key)


def _extract_user_rate_limit_key(request: Request) -> str | None:
    if not use_supabase():
        return None
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        payload = _decode_and_verify_supabase_jwt(token)
        request.state.bearer_payload = payload
    except HTTPException:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return f"user:{user_id}"


def _rate_limit_key(request: Request) -> str:
    user_key = _extract_user_rate_limit_key(request)
    if user_key:
        return user_key
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _record_error_event(
    *,
    request: Request,
    status: int,
    error_code: str,
    message: str,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": request.url.path,
        "method": request.method,
        "status": int(status),
        "error_code": error_code,
        "message": (message or "")[:500],
    }
    _error_events.append(event)
    logger.error("ERROR_EVENT %s", json.dumps(event, sort_keys=True))


origins = resolve_allowed_origins(os.getenv("NEXUS_ALLOWED_ORIGINS"), os.getenv("NEXUS_ENV"))
logger.info("CORS allowed origins: %s", ", ".join(origins) if origins else "(none)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path
    limiter_key = _rate_limit_key(request)
    limiter_scope = "user" if limiter_key.startswith("user:") else "ip"
    if RATE_LIMIT_ENABLED and path not in RATE_LIMIT_EXEMPT_PATHS:
        allowed, retry_after, remaining = _rate_limiter.allow(limiter_key, time.time())
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please retry shortly.",
                        "hint": "Wait before retrying or reduce request frequency.",
                    }
                },
            )
            response.headers["Retry-After"] = str(retry_after)
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_WINDOW)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW_SECONDS)
            response.headers["X-RateLimit-Scope"] = limiter_scope
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            return response
    else:
        remaining = RATE_LIMIT_PER_WINDOW

    try:
        response = await call_next(request)
    except Exception as exc:
        _record_error_event(
            request=request,
            status=500,
            error_code="UNHANDLED_EXCEPTION",
            message=str(exc),
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_WINDOW)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW_SECONDS)
    response.headers["X-RateLimit-Scope"] = limiter_scope
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    if response.status_code >= 500:
        _record_error_event(
            request=request,
            status=response.status_code,
            error_code="HTTP_5XX",
            message=f"HTTP {response.status_code}",
        )
    if TIMING_LOG_ENABLED:
        logger.info(
            "HTTP_REQUEST method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            path,
            response.status_code,
            duration_ms,
        )
    return response

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

def _load_manifest(run_id: str, user_id: str | None = None) -> dict:
    manifest_path = ROOT / "data" / "cache" / "manifests" / f"{run_id}.json"
    if manifest_path.exists():
        return _load_json(manifest_path)
    run = repo.get_run_by_id(run_id, user_id=user_id if use_supabase() else None)
    if run and run.manifest_json:
        try:
            return json.loads(run.manifest_json)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Manifest not found")

def _load_summary(run_id: str, user_id: str | None = None) -> dict:
    summary_path = EXPORTS_DIR / run_id / f"summary_{run_id}.json"
    if summary_path.exists():
        return _load_json(summary_path)
    summary_path = EXPORTS_DIR / run_id / "summary.json"
    if summary_path.exists():
        return _load_json(summary_path)
    if use_supabase():
        return _load_supabase_json(run_id, "summary.json", user_id=user_id)
    return _load_json(summary_path)

def _load_performance(run_id: str, user_id: str | None = None) -> list[dict]:
    perf_path = EXPORTS_DIR / run_id / "performance.csv"
    if perf_path.exists():
        return _load_csv(perf_path)
    if use_supabase():
        return _load_supabase_csv(run_id, "performance.csv", user_id=user_id)
    return _load_csv(perf_path)

def _load_monthly_returns(run_id: str, user_id: str | None = None) -> list[dict]:
    returns_path = EXPORTS_DIR / run_id / "monthly_returns.csv"
    if returns_path.exists():
        return _load_csv(returns_path)
    if use_supabase():
        return _load_supabase_csv(run_id, "monthly_returns.csv", user_id=user_id)
    return _load_csv(returns_path)

def _load_attribution_summary(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "attribution_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "attribution_summary.json", user_id=user_id)
    raise HTTPException(status_code=404, detail="Attribution summary not found")


def _load_attribution_timeseries(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "attribution_timeseries.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "attribution_timeseries.csv", user_id=user_id)
    return []


def _load_risk_contribution(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "risk_contribution.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "risk_contribution.json", user_id=user_id)
    raise HTTPException(status_code=404, detail="Risk contribution not found")


def _load_macro_regimes(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "macro_regime_flags.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "macro_regime_flags.csv", user_id=user_id)
    return []


def _load_macro_summary(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "macro_regime_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "macro_regime_summary.json", user_id=user_id)
    return {}


def _load_macro_context(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "macro_context.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "macro_context.json", user_id=user_id)
    return {}


def _load_coverage_summary(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "coverage_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "coverage_summary.json", user_id=user_id)
    return {}

def _load_risk_free_series(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "risk_free_series.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "risk_free_series.csv", user_id=user_id)
    return []

def _load_corporate_actions(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "corporate_actions_events.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "corporate_actions_events.csv", user_id=user_id)
    return []

def _load_data_contracts(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "data_contracts.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "data_contracts.json", user_id=user_id)
    return {}


def _load_rolling_metrics(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "rolling_metrics.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "rolling_metrics.csv", user_id=user_id)
    return []


def _load_benchmark_comparison(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "benchmark_comparison.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "benchmark_comparison.json", user_id=user_id)
    raise HTTPException(status_code=404, detail="Benchmark comparison not found")


def _load_benchmark_timeseries(run_id: str, user_id: str | None = None) -> list[dict]:
    path = EXPORTS_DIR / run_id / "benchmark_timeseries.csv"
    if path.exists():
        return _load_csv(path)
    if use_supabase():
        return _load_supabase_csv(run_id, "benchmark_timeseries.csv", user_id=user_id)
    return []


def _load_concentration_summary(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "concentration_summary.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "concentration_summary.json", user_id=user_id)
    return {}


def _load_factor_tilts(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "factor_tilts.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "factor_tilts.json", user_id=user_id)
    return {}


def _load_report_html(run_id: str, user_id: str | None = None) -> str:
    report_path = EXPORTS_DIR / run_id / "report.html"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8")
    if use_supabase():
        try:
            data, _ = repo.get_artifact_bytes(run_id, "report.html", user_id=user_id)
            return data.decode("utf-8")
        except FileNotFoundError:
            pass
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load report HTML: {exc}")
    raise HTTPException(status_code=404, detail="Report not found. Run compute to generate report.html.")


def _load_diagnostics(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "diagnostics.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "diagnostics.json", user_id=user_id)
    return {"diagnostics": [], "run_id": run_id}


def _load_correlation_matrix(run_id: str, user_id: str | None = None) -> dict:
    path = EXPORTS_DIR / run_id / "correlation_matrix.json"
    if path.exists():
        return _load_json(path)
    if use_supabase():
        return _load_supabase_json(run_id, "correlation_matrix.json", user_id=user_id)
    return {"status": "unavailable", "matrix": {}}

def _load_supabase_json(run_id: str, filename: str, user_id: str | None = None) -> dict:
    try:
        data, _content_type = repo.get_artifact_bytes(run_id, filename, user_id=user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load artifact: {exc}")

def _load_supabase_csv(run_id: str, filename: str, user_id: str | None = None) -> list[dict]:
    try:
        data, _content_type = repo.get_artifact_bytes(run_id, filename, user_id=user_id)
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

def _load_portfolio(portfolio_id: int, user_id: str | None = None) -> dict:
    if use_supabase():
        from storage_supabase.db import session_scope as supa_session_scope
        from storage_supabase import models as supa_models

        with supa_session_scope() as session:
            try:
                query = session.query(supa_models.Portfolio).filter_by(id=portfolio_id)
                if user_id is not None:
                    query = query.filter_by(user_id=str(user_id))
                portfolio = query.first()
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


def _assert_portfolio_access(portfolio_id: int, current_user: dict) -> None:
    if not use_supabase():
        return
    user_id = current_user.get("user_id")
    _load_portfolio(portfolio_id, user_id=str(user_id) if user_id is not None else None)


def _assert_run_access(run_id: str, current_user: dict) -> str | None:
    if not use_supabase():
        return None
    user_id = current_user.get("user_id")
    resolved_user_id = str(user_id) if user_id is not None else None
    if resolved_user_id is None:
        raise HTTPException(status_code=401, detail="Missing authenticated user.")
    run = repo.get_run_by_id(run_id, user_id=resolved_user_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return resolved_user_id

def _load_ui_state() -> dict:
    path = ROOT / "data" / "user_uploads" / "ui_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _resolve_portfolio_id(portfolio_id: str, user_id: str | int | None = None) -> int:
    if portfolio_id in ("default", "main", "primary"):
        if use_supabase() and user_id is None:
            raise HTTPException(status_code=401, detail="Missing authenticated user context.")
        resolved_user_id = user_id if user_id is not None else data_manager.get_current_user_id()
        return data_manager.get_main_portfolio_id(resolved_user_id)
    try:
        return int(portfolio_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid portfolio id")


def _validate_portfolio_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Portfolio name is required.")
    if len(cleaned) > 80:
        raise HTTPException(status_code=400, detail="Portfolio name must be 80 characters or fewer.")
    return cleaned


def _validate_currency(code: str | None) -> str:
    cleaned = (code or "USD").strip().upper()
    if len(cleaned) != 3 or not cleaned.isalpha():
        raise HTTPException(status_code=400, detail="Currency must be a 3-letter ISO code (e.g., USD).")
    return cleaned

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


def _current_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


@app.get("/ops/health")
def ops_health(full: bool = False):
    uptime_seconds = int((datetime.now(timezone.utc) - APP_STARTED_AT).total_seconds())
    runtime = {
        "uptime_seconds": uptime_seconds,
        "started_at": APP_STARTED_AT.isoformat(),
        "now": datetime.now(timezone.utc).isoformat(),
        "rss_mb": _current_rss_mb(),
    }

    database = {
        "status": "unknown",
        "backend": "supabase" if use_supabase() else "sqlite",
    }
    try:
        latest = repo.get_latest_run()
        database["status"] = "connected"
    except Exception:
        latest = None
        database["status"] = "disconnected"

    if full:
        cache_stats = _ttl_cache.get("ops_cache_stats", OPS_CACHE_STATS_TTL_SECONDS)
        if cache_stats is None:
            try:
                from src.services.cache_service import CacheService

                cache_stats = CacheService().get_cache_stats()
                _ttl_cache.set("ops_cache_stats", cache_stats)
            except Exception:
                cache_stats = {}
    else:
        cache_stats = {"status": "skipped", "reason": "Use full=true to include cache stats."}

    latest_run = None
    if latest:
        latest_run_id = getattr(latest, "run_id", None) or getattr(latest, "id", None)
        latest_run = {
            "run_id": latest_run_id,
            "status": getattr(latest, "status", None),
            "timestamp": latest.created_at.isoformat() if getattr(latest, "created_at", None) else None,
        }

    return {
        "status": "ok",
        "runtime": runtime,
        "database": database,
        "cache": cache_stats,
        "rate_limit": {
            "enabled": RATE_LIMIT_ENABLED,
            "limit_per_window": RATE_LIMIT_PER_WINDOW,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
        "latest_run": latest_run,
    }

@app.post("/run")
async def create_run(
    run_type: str = Form("uploaded"),
    portfolio_id: str = Form("default"),
    file: UploadFile | None = File(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new portfolio run from an uploaded trades CSV or demo data.
    """
    resolved_portfolio_id = _resolve_portfolio_id(portfolio_id, user_id=current_user.get("user_id"))
    _assert_portfolio_access(resolved_portfolio_id, current_user)
    logger.info("RUN_START run_type=%s portfolio_id=%s file=%s", run_type, resolved_portfolio_id, bool(file))
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
            logger.exception("RUN_COMPUTE_FAILED: Full traceback for demo run_id=%s", run_id)
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

    # Validate file size to prevent OOM
    row_count = len(df)
    MAX_ROWS_WARNING = 500
    MAX_ROWS_HARD_LIMIT = 2000

    if row_count > MAX_ROWS_HARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "FILE_TOO_LARGE",
                "message": f"CSV file too large ({row_count} rows).",
                "details": {"row_count": row_count, "max_rows": MAX_ROWS_HARD_LIMIT},
                "hint": f"Please reduce file size to under {MAX_ROWS_HARD_LIMIT} rows or split into multiple uploads.",
            },
        )

    if row_count > MAX_ROWS_WARNING:
        logger.warning(
            "LARGE_FILE_UPLOAD row_count=%s warning_threshold=%s",
            row_count,
            MAX_ROWS_WARNING,
        )

    # Use PortfolioService for validation
    portfolio_service = PortfolioService()
    validated, errors = portfolio_service.validate_and_prepare_ledger(df)
    
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

    logger.info("RUN_INPUT rows=%s cols=%s", len(validated), list(validated.columns))
    data_manager.save_portfolio_inputs(resolved_portfolio_id, validated, None)

    tickers = portfolio_service.extract_tickers(validated)
    logger.info("RUN_TICKERS count=%s tickers=%s", len(tickers), tickers[:10] if len(tickers) > 10 else tickers)
    log_rss("after_csv_read")  # Track memory after CSV parsing

    # Market data is fetched inside compute_app_state → get_prices()
    # No separate pre-fetch needed — eliminates redundant fetching that doubled processing time

    log_rss("before_compute")  # Track memory before computation
    
    # Aggressive cleanup before compute to reduce memory footprint  
    gc.collect()
    
    logger.info("RUN_COMPUTE_START run_id=%s portfolio_id=%s", run_id, resolved_portfolio_id)
    try:
        app_state = compute_app_state(
            portfolio_id=resolved_portfolio_id,
            run_id=run_id,
            save_run=True,
            source_override="Ledger",
            uploads_active=True,
            run_type="uploaded",
            trade_tickers=tickers,
        )
        logger.info("RUN_COMPUTE_SUCCESS run_id=%s", run_id)
        log_rss("after_compute")  # Track memory after computation
        save_artifacts(app_state)
        logger.info("RUN_ARTIFACTS_SAVED run_id=%s", run_id)
        manifest = app_state.run_manifest
        
        # Build response with warnings if any tickers had missing market data
        response = {
            "run_id": manifest.run_id if manifest else "",
            "status": "completed",
            "timestamp": manifest.timestamp if manifest else None,
        }
        
        # Surface failed tickers from the pipeline's price metadata
        failed_tickers = app_state.price_meta.missing_tickers if app_state.price_meta else []
        if failed_tickers:
            logger.warning("RUN_TICKERS_MISSING count=%s tickers=%s", len(failed_tickers), failed_tickers)
            response["warnings"] = {
                "failed_tickers": {
                    "count": len(failed_tickers),
                    "tickers": failed_tickers,
                    "message": f"{len(failed_tickers)} ticker(s) failed to load market data. Results may be incomplete.",
                }
            }
        
        return response
    except Exception as exc:
        logger.exception("RUN_COMPUTE_FAILED: Full traceback for run_id=%s", run_id)
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

@app.get("/admin/cache-status")
def get_cache_status(x_admin_key: str | None = Header(default=None)):
    """
    Diagnostic endpoint to check cache index health and local storage size.
    """
    if not _admin_key_valid(x_admin_key):
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 1. Check Supabase / DB Connection and Index Count
    db_status = "unknown"
    index_count = 0
    try:
        if use_supabase():
            from storage_supabase.db import session_scope
            from storage_supabase import models
        else:
            from storage.db import session_scope
            from storage import models

        with session_scope() as session:
            index_count = session.query(models.DataCacheIndex).count()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    # 2. Check Local Cache Size
    cache_dir = ROOT / "data" / "market_cache" / "persistent"
    total_size_mb = 0.0
    file_count = 0
    if cache_dir.exists():
        for p in cache_dir.rglob("*"):
            if p.is_file():
                total_size_mb += p.stat().st_size
                file_count += 1
    total_size_mb /= (1024 * 1024)

    # 3. Overall Status
    status = "ok"
    if db_status != "connected":
        status = "degraded"
    
    return {
        "status": status,
        "database": db_status,
        "driver": "supabase" if use_supabase() else "sqlite",
        "cache_index_entries": index_count,
        "local_cache_size_mb": round(total_size_mb, 2),
        "local_cache_files": file_count,
        "local_cache_path": str(cache_dir)
    }


@app.delete("/admin/clear-cache")
def clear_cache(x_admin_key: str | None = Header(default=None)):
    """
    Admin endpoint to clear stale market data cache.
    Use when Yahoo Finance data is returning old/stale dates.
    Clears both local filesystem and Supabase storage.
    """
    if not _admin_key_valid(x_admin_key):
        raise HTTPException(status_code=403, detail="Unauthorized")

    result = {
        "status": "cleared",
        "local_files_deleted": 0,
        "local_bytes_freed": 0,
        "supabase_files_deleted": 0,
        "cache_index_cleared": 0,
    }

    # 1. Clear local filesystem cache
    cache_dir = ROOT / "data" / "market_cache" / "persistent" / "yahoo"
    if cache_dir.exists():
        for p in cache_dir.glob("*.parquet"):
            if p.is_file():
                result["local_bytes_freed"] += p.stat().st_size
                p.unlink()
                result["local_files_deleted"] += 1
    
    # 2. Clear Supabase storage and cache index if using Supabase
    if use_supabase():
        try:
            from storage_supabase.storage import list_files, delete_file
            from storage_supabase.db import session_scope
            from storage_supabase import models
            
            bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "nexus-artifacts")
            
            # Delete files from Supabase storage bucket
            try:
                files = list_files(bucket, "cache/yahoo/")
                for file_info in files:
                    try:
                        delete_file(bucket, f"cache/yahoo/{file_info['name']}")
                        result["supabase_files_deleted"] += 1
                    except Exception as e:
                        logger.warning("SUPABASE_DELETE_FAILED file=%s error=%s", file_info.get('name'), e)
            except Exception as e:
                logger.warning("SUPABASE_LIST_FAILED error=%s", e)
            
            # Clear cache index entries for yahoo source
            with session_scope() as session:
                deleted = session.query(models.DataCacheIndex).filter(
                    models.DataCacheIndex.source == "yahoo"
                ).delete()
                session.commit()
                result["cache_index_cleared"] = deleted
                
        except Exception as e:
            logger.error("SUPABASE_CLEAR_FAILED error=%s", e)
            result["supabase_error"] = str(e)
    
    logger.info("CACHE_CLEARED result=%s", result)
    return result


@app.get("/admin/error-events")
def get_error_events(
    x_admin_key: str | None = Header(default=None),
    status_min: int = 500,
    limit: int = 20,
):
    if not _admin_key_valid(x_admin_key):
        raise HTTPException(status_code=403, detail="Unauthorized")
    resolved_limit = max(1, min(int(limit), 200))
    resolved_status_min = max(100, int(status_min))
    filtered = [
        event
        for event in reversed(list(_error_events))
        if int(event.get("status", 0)) >= resolved_status_min
    ][:resolved_limit]
    return {
        "events": filtered,
        "count": len(filtered),
        "buffer_size": len(_error_events),
        "max_buffer_size": max(10, ERROR_EVENTS_MAX),
    }

@app.post("/api/v1/run")
async def create_run_alias(
    run_type: str = Form("uploaded"),
    portfolio_id: str = Form("default"),
    file: UploadFile | None = File(None),
    current_user: dict = Depends(get_current_user),
):
    return await create_run(
        run_type=run_type,
        portfolio_id=portfolio_id,
        file=file,
        current_user=current_user,
    )

@app.get("/runs")
def list_runs(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user.get("user_id")) if use_supabase() else None
    runs = [run for run in repo.list_runs(user_id=user_id) if run.status == "completed"]
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
def list_runs_alias(current_user: dict = Depends(get_current_user)):
    return list_runs(current_user=current_user)

@app.get("/api/v1/run/latest")
def get_latest_run(current_user: dict = Depends(get_current_user)):
    """
    Get metadata for the last completed run.
    """
    try:
        user_id = str(current_user.get("user_id")) if use_supabase() else None
        last_run = repo.get_latest_run(user_id=user_id)
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
def latest_run_alias(current_user: dict = Depends(get_current_user)):
    return get_latest_run(current_user=current_user)

@app.get("/api/v1/run/{run_id}/summary")
def get_run_summary(run_id: str, current_user: dict = Depends(get_current_user)):
    """
    Serve the precomputed JSON summary artifact.
    """
    # Security: Validate run_id format to prevent traversal (basic)
    _validate_run_id(run_id)
    user_id = _assert_run_access(run_id, current_user)
    return _load_summary(run_id, user_id=user_id)

@app.get("/api/v1/run/{run_id}")
def get_run(run_id: str, current_user: dict = Depends(get_current_user)):
    _validate_run_id(run_id)
    user_id = _assert_run_access(run_id, current_user)
    manifest = _load_manifest(run_id, user_id=user_id)
    try:
        summary = _load_summary(run_id, user_id=user_id)
    except HTTPException:
        summary = _empty_summary("Summary artifact not available.")
    try:
        performance = _load_performance(run_id, user_id=user_id)
    except HTTPException:
        performance = []
    try:
        monthly_returns = _load_monthly_returns(run_id, user_id=user_id)
    except HTTPException:
        monthly_returns = []
    try:
        attribution_summary = _load_attribution_summary(run_id, user_id=user_id)
    except HTTPException:
        attribution_summary = {}
    attribution_timeseries = _load_attribution_timeseries(run_id, user_id=user_id)
    try:
        risk_contribution = _load_risk_contribution(run_id, user_id=user_id)
    except HTTPException:
        risk_contribution = {"summary": {}, "contributions": []}
    rolling_metrics = _load_rolling_metrics(run_id, user_id=user_id)
    macro_regimes = _load_macro_regimes(run_id, user_id=user_id)
    macro_summary = _load_macro_summary(run_id, user_id=user_id)
    macro_context = _load_macro_context(run_id, user_id=user_id)
    try:
        benchmark_comparison = _load_benchmark_comparison(run_id, user_id=user_id)
    except HTTPException:
        benchmark_comparison = {}
    benchmark_timeseries = _load_benchmark_timeseries(run_id, user_id=user_id)
    concentration_summary = _load_concentration_summary(run_id, user_id=user_id)
    factor_tilts = _load_factor_tilts(run_id, user_id=user_id)
    diagnostics_payload = _load_diagnostics(run_id, user_id=user_id)
    correlation_matrix = _load_correlation_matrix(run_id, user_id=user_id)
    coverage_summary = _load_coverage_summary(run_id, user_id=user_id)
    risk_free_series = _load_risk_free_series(run_id, user_id=user_id)
    corporate_actions = _load_corporate_actions(run_id, user_id=user_id)
    data_contracts = _load_data_contracts(run_id, user_id=user_id)
    risk = _compute_risk_metrics(performance, risk_free_series)
    equity_curve = [
        {"date": row.get("date"), "value": row.get("value")}
        for row in performance
        if row.get("date") is not None and row.get("value") is not None
    ]
    metric_payload = {
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
        "concentration": concentration_summary,
        "factor_tilts": factor_tilts,
        "diagnostics": diagnostics_payload.get("diagnostics", []),
        "correlation_matrix": correlation_matrix,
    }
    assert_run_metric_payload_keys_registered(metric_payload.keys())

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
        **metric_payload,
    }

@app.get("/run/{run_id}")
def run_alias(run_id: str, current_user: dict = Depends(get_current_user)):
    return get_run(run_id, current_user=current_user)


@app.get("/api/reports/{run_id}/pdf")
def get_report_pdf(run_id: str, current_user: dict = Depends(get_current_user)):
    _validate_run_id(run_id)
    user_id = _assert_run_access(run_id, current_user)
    html = _load_report_html(run_id, user_id=user_id)
    try:
        pdf_bytes = html_to_pdf_bytes(html, base_url=str((EXPORTS_DIR / run_id).resolve()))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    filename = f"report_{run_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/api/v1/reports/{run_id}/pdf")
def get_report_pdf_alias_v1(run_id: str, current_user: dict = Depends(get_current_user)):
    return get_report_pdf(run_id, current_user=current_user)

@app.get("/api/v1/portfolio/{portfolio_id}")
def get_portfolio(portfolio_id: str, current_user: dict = Depends(get_current_user)):
    resolved_id = _resolve_portfolio_id(portfolio_id, user_id=current_user.get("user_id"))
    auth_user_id = current_user.get("user_id")
    portfolio = _load_portfolio(resolved_id, user_id=str(auth_user_id) if auth_user_id is not None else None)
    holdings = _attach_prices(_load_holdings(resolved_id))
    ui_state = _load_ui_state()
    portfolio["benchmark"] = ui_state.get("benchmark")
    return {"portfolio": portfolio, "holdings": holdings}

@app.get("/portfolio/{portfolio_id}")
def portfolio_alias(portfolio_id: str, current_user: dict = Depends(get_current_user)):
    return get_portfolio(portfolio_id, current_user=current_user)


@app.post("/api/v1/portfolios")
def create_portfolio(payload: dict, current_user: dict = Depends(get_current_user)):
    name = _validate_portfolio_name(str(payload.get("name", "")))
    currency = _validate_currency(payload.get("currency"))

    if use_supabase():
        from storage_supabase.db import session_scope as supa_session_scope
        from storage_supabase import models as supa_models

        user_id = current_user.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Missing authenticated user context.")
        with supa_session_scope() as session:
            portfolio = supa_models.Portfolio(
                user_id=str(user_id),
                name=name,
                base_currency=currency,
            )
            session.add(portfolio)
            session.flush()
            return {
                "portfolio": {
                    "id": portfolio.id,
                    "name": portfolio.name,
                    "currency": portfolio.base_currency,
                }
            }

    with session_scope() as session:
        portfolio = Portfolio(
            user_id=0,
            name=name,
            currency=currency,
        )
        session.add(portfolio)
        session.flush()
        return {
            "portfolio": {
                "id": portfolio.id,
                "name": portfolio.name,
                "currency": getattr(portfolio, "currency", currency),
            }
        }

@app.get("/api/v1/definitions")
def get_definitions():
    cached = _ttl_cache.get("definitions_registry", DEFINITIONS_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    payload = get_exposed_definitions()
    _ttl_cache.set("definitions_registry", payload)
    return payload

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
    if not _admin_key_valid(x_admin_key):
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
def export_artifact(run_id: str, artifact: str, current_user: dict = Depends(get_current_user)):
    _validate_run_id(run_id)
    user_id = _assert_run_access(run_id, current_user)
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
        "concentration-summary": "concentration_summary.json",
        "factor-tilts": "factor_tilts.json",
        "diagnostics": "diagnostics.json",
        "correlation-matrix": "correlation_matrix.json",
    }
    assert_metric_artifact_aliases_registered(
        {
            "attribution-summary",
            "attribution-timeseries",
            "risk-contribution",
            "risk-contribution-json",
            "macro-regimes",
            "macro-regime-summary",
            "macro-context",
            "rolling-metrics",
            "benchmark-comparison",
            "benchmark-timeseries",
            "concentration-summary",
            "factor-tilts",
            "diagnostics",
            "correlation-matrix",
        }
    )
    filename = allowed.get(artifact)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    if use_supabase():
        try:
            data, content_type = repo.get_artifact_bytes(run_id, filename, user_id=user_id)
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
def export_artifact_alias(run_id: str, artifact: str, current_user: dict = Depends(get_current_user)):
    return export_artifact(run_id, artifact, current_user=current_user)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# =====================================================
# Market Data Refresh Admin Endpoints
# =====================================================

@app.post("/admin/refresh-market-data")
async def refresh_market_data(days_back: int = 30, force_refresh: bool = False):
    """Manually trigger market data refresh for all active tickers."""
    from src.services.market_data_refresh_service import MarketDataRefreshService
    
    service = MarketDataRefreshService()
    tickers = service.get_active_tickers()
    
    if not tickers:
        return {
            "status": "no_tickers",
            "message": "No active tickers found in portfolios"
        }
    
    results = service.refresh_market_data(tickers, days_back, force_refresh)
    validation = service.validate_refresh_results(results)
    
    return {
        "status": "completed",
        "results": results,
        "validation": validation
    }


@app.get("/admin/refresh-status")
async def get_refresh_status():
    """Get basic statistics about cached market data."""
    from src.services.cache_service import CacheService
    
    cache_service = CacheService()
    stats = cache_service.get_cache_stats()
    
    return {
        "status": "ok",
        "cache_stats": stats
    }


@app.post("/admin/backfill-market-data")
async def backfill_market_data(request: dict):
    """Backfill market data for specific tickers and date range."""
    from datetime import datetime
    from market_data.store import MarketDataStore
    
    # Extract from request body
    tickers = request.get("tickers", [])
    start_date = request.get("start_date")
    end_date = request.get("end_date")
    
    if not tickers or not start_date or not end_date:
        raise HTTPException(
            status_code=400, 
            detail="Missing required fields: tickers, start_date, end_date"
        )
    
    # Validate dates
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Validate date range (max 1 year)
    if (end - start).days > 365:
        raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year")
    
    if end < start:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")
    
    # Fetch data
    store = MarketDataStore.default()
    success_count = 0
    failure_count = 0
    errors = []
    
    for ticker in tickers:
        try:
            prices = store.get_prices(ticker, start_date, end_date)
            if not prices.empty:
                success_count += 1
            else:
                failure_count += 1
                errors.append({"ticker": ticker, "error": "No data returned"})
        except Exception as e:
            failure_count += 1
            errors.append({"ticker": ticker, "error": str(e)})
    
    return {
        "status": "completed",
        "success_count": success_count,
        "failure_count": failure_count,
        "total_tickers": len(tickers),
        "errors": errors,
        "date_range": {
            "start": start_date,
            "end": end_date
        }
    }
