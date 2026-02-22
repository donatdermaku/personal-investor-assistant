import uuid
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.app_state import AppState
from src.manifest import create_manifest, RunManifest, compute_input_hash
from src.coverage import build_coverage_summary
from src.risk_free import compute_risk_free_series
from src.utils_io import ROOT
from src.portfolio import load_portfolio, PortfolioResult
from src.streamlit_data import (
    get_prices, get_scores, get_fundamentals,
    market_status, get_benchmark_prices,
)
from storage.datamanager import data_manager
from storage import repo
from storage.db import session_scope

# Define standard export paths relative to ROOT/data
EXPORTS_DIR_ENV = os.getenv("NEXUS_EXPORT_DIR", "data/exports")
EXPORTS_DIR = Path(EXPORTS_DIR_ENV)
if not EXPORTS_DIR.is_absolute():
    EXPORTS_DIR = (ROOT / EXPORTS_DIR).resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_app_state(
    portfolio_id: int | None = None,
    run_id: str | None = None,
    save_run: bool = True,
    source_override: str | None = None,
    uploads_active: bool = True,
    run_type: str | None = None,
    trade_tickers: list[str] | None = None,
) -> AppState:
    """
    Central logic to compute the application state.
    - Loads user inputs (DB/File)
    - Loads market data (DuckDB)
    - Computes PortfolioResult
    - Generates RunManifest
    - If save_run=True: Persists Run and Artifacts to DB
    """
    
    # 1. Resolve User/Portfolio
    if portfolio_id is None:
        user_id = data_manager.get_current_user_id()
        portfolio_id = data_manager.get_main_portfolio_id(user_id)
    
    # 2. Check Market Status
    status_str, market_state = market_status()
    
    # 3. Initialize Run (if saving)
    run_id = run_id or str(uuid.uuid4())
    
    if save_run:
        repo.create_run(run_id, portfolio_id, None, None, run_type=run_type)
    
    # 4. Load User Data (Watchlist)
    watch_tickers = data_manager.load_watchlist() or []
    
    # 5. Determine tickers to use for this run
    # Prioritize trade_tickers from CSV upload over watchlist
    coverage_tickers = trade_tickers if trade_tickers is not None else watch_tickers
    
    # 6. Load Market Data
    # Use coverage_tickers to ensure we load prices for actual trade tickers
    prices, price_meta = get_prices(market_state, coverage_tickers)
    scores, scores_meta = get_scores(coverage_tickers)
    fund, fund_meta = get_fundamentals(coverage_tickers)
    try:
        from src.utils_memory import log_rss
        log_rss("after_market_data")
    except Exception:
        pass
    
    # 7. Load Portfolio
    # Here we use the underlying 'load_portfolio' which now uses DataManager
    # Note: load_portfolio calls DataManager internally for trades/snapshot.
    portfolio_result = load_portfolio(
        prices,
        coverage_tickers,
        source_override=source_override,
        uploads_active=uploads_active,
        portfolio_id=portfolio_id,
    )
    try:
        from src.utils_memory import log_rss
        log_rss("after_portfolio")
    except Exception:
        pass
    
    # 8. Benchmark
    bench_ticker = "SPY" # Default, should come from settings
    bench_prices, bench_meta = get_benchmark_prices(bench_ticker)
    
    # 9. Create Manifest
    # Calculate hashes for manifest
    # We need raw inputs for input_hash. 'load_portfolio' consumes them but doesn't return raw easily.
    # We can fetch them again for hashing or trust that RunManifest handles it?
    # RunManifest logic in P7 expects "PortfolioInput" objects or similar.
    # For P11, let's keep it simple: generic manifest creation.
    
    # We really want to version the *inputs*.
    trades_df = data_manager.load_trades(portfolio_id)
    snapshot_df = data_manager.load_snapshot(portfolio_id)
    
    # Compute Input Hash manually or via helper
    input_str = ""
    if not trades_df.empty:
        input_str += str(pd.util.hash_pandas_object(trades_df).sum())
    if not snapshot_df.empty:
        input_str += str(pd.util.hash_pandas_object(snapshot_df).sum())
    input_hash = compute_input_hash(input_str)
    
    as_of = portfolio_result.daily_values.index.max().strftime("%Y-%m-%d") if not portfolio_result.daily_values.empty else None
    risk_free = compute_risk_free_series(portfolio_result.daily_values.index if not portfolio_result.daily_values.empty else pd.DatetimeIndex([]))
    from src.analytics.required_start import compute_required_start_per_ticker
    req_starts = compute_required_start_per_ticker(portfolio_result.holdings_daily)
    
    # CRITICAL FIX: Use coverage_tickers (trade tickers) instead of watch_tickers
    coverage_summary = build_coverage_summary(
        prices,
        required_tickers=coverage_tickers,  # ← FIXED: Use trade tickers, not empty watchlist
        benchmark_ticker=bench_ticker,
        benchmark_prices=bench_prices,
        as_of=as_of,
        risk_free_series=risk_free.series,
        required_start_per_ticker=req_starts,
    )
    try:
        from src.utils_memory import log_rss
        log_rss("after_coverage")
    except Exception:
        pass

    manifest = create_manifest(
        run_id=run_id,
        input_hash=input_hash,
        config_hash="default", # Placeholder
        market_data_hash="duckdb_latest", # Placeholder
        portfolio_result=portfolio_result,
        coverage_summary=coverage_summary,
    )

    # 10. Persist latest holdings snapshot (if available)
    _persist_snapshot_from_result(portfolio_id, portfolio_result)

    # 11. Update Run in DB
    if save_run:
        repo.update_run_complete(run_id, manifest.to_json(), run_type=run_type)
    
    # 12. Assemble AppState
    app_state = AppState(
        run_manifest=manifest,
        portfolio=portfolio_result,
        prices=prices,
        scores=scores,
        watch_tickers=coverage_tickers,  # Use the tickers that were actually used
        price_meta=price_meta,
        fundamentals_meta=fund_meta,
        scores_meta=scores_meta,
        benchmark_prices=bench_prices,
        risk_free=risk_free,
        market_state=market_state
    )
    
    return app_state


def _persist_snapshot_from_result(portfolio_id: int, result: PortfolioResult) -> None:
    if result.holdings_daily.empty:
        return
    snapshot = result.holdings_daily.copy()
    if "date" in snapshot.columns:
        latest_date = snapshot["date"].max()
        snapshot = snapshot[snapshot["date"] == latest_date]
    if "quantity" not in snapshot.columns and "shares" in snapshot.columns:
        snapshot = snapshot.rename(columns={"shares": "quantity"})
    snapshot = snapshot[["ticker", "quantity"]].copy()
    snapshot["quantity"] = pd.to_numeric(snapshot["quantity"], errors="coerce").fillna(0.0)
    snapshot = snapshot[snapshot["quantity"] > 0]
    if snapshot.empty:
        return
    data_manager.save_portfolio_inputs(portfolio_id, trades=None, snapshot=snapshot)

def save_artifacts(app_state: AppState):
    """
    Generate and save export files (HTML, JSON, CSV).
    Record them in DB.
    """
    from src.streamlit_export import (
        export_summary_json, 
        export_performance_csv, 
        export_monthly_returns_csv, 
        save_html_report,
        export_coverage_summary_json,
        export_attribution_summary_json,
        export_attribution_timeseries_csv,
        export_risk_contribution_csv,
        export_risk_contribution_json,
        export_macro_regime_flags_csv,
        export_macro_regime_summary_json,
        export_macro_context_json,
        export_rolling_metrics_csv,
        export_benchmark_comparison_json,
        export_concentration_summary_json,
        export_factor_tilts_json,
        export_benchmark_timeseries_csv,
        export_risk_free_series_csv,
        export_corporate_actions_csv,
        export_data_contracts_json,
        export_diagnostics_json,
        export_correlation_matrix_json,
    )
    from src.analytics.attribution import compute_attribution
    from src.analytics.concentration import build_hhi_summary
    from src.analytics.factors import compute_factor_tilts
    from src.analytics.metrics_registry import assert_metrics_registered
    from src.analytics.risk import compute_risk_contributions_from_cov
    from src.analytics.rolling import compute_rolling_metrics
    from src.analytics.correlation import compute_correlation_matrix_from_cov
    from src.analytics.streaming import build_canonical_calendar, iter_price_state, OnlineCovariance
    from src.analytics.comparative import compute_benchmark_comparison
    from src.analytics.macro import compute_macro_regime_payload
    from market_data.fred import get_cached_series
    from src.diagnostics.engine import diagnostics_payload, generate_diagnostics
    from market_data.contracts import contract_registry
    
    manifest = app_state.run_manifest
    if not manifest:
        return
        
    run_id = manifest.run_id
    base_path = EXPORTS_DIR / run_id
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Json Summary
    json_path = base_path / "summary.json"
    export_summary_json(json_path, app_state.portfolio, manifest)
    repo.add_artifact(run_id, "summary_json", str(json_path))

    coverage_path = base_path / "coverage_summary.json"
    export_coverage_summary_json(coverage_path, manifest.coverage_summary)
    repo.add_artifact(run_id, "coverage_summary_json", str(coverage_path))

    risk_free_path = base_path / "risk_free_series.csv"
    export_risk_free_series_csv(risk_free_path, app_state.risk_free.series)
    repo.add_artifact(run_id, "risk_free_series_csv", str(risk_free_path))

    contracts_path = base_path / "data_contracts.json"
    export_data_contracts_json(contracts_path, contract_registry())
    repo.add_artifact(run_id, "data_contracts_json", str(contracts_path))

    if not app_state.prices.empty and {"dividend", "split_ratio"}.issubset(app_state.prices.columns):
        events = app_state.prices.copy()
        events = events[(events["dividend"] > 0) | (events["split_ratio"] != 1.0)]
        events = events[["date", "ticker", "dividend", "split_ratio"]].copy()
        actions_path = base_path / "corporate_actions_events.csv"
        export_corporate_actions_csv(actions_path, events)
        repo.add_artifact(run_id, "corporate_actions_events_csv", str(actions_path))
    
    # HTML Report
    html_path = base_path / "report.html"
    save_html_report(html_path, app_state)
    repo.add_artifact(run_id, "html_report", str(html_path))
    
    performance_records: list[dict] = []
    rolling_records: list[dict] = []
    # CSVs
    if not app_state.portfolio.daily_values.empty:
        perf_path = base_path / "performance.csv"
        export_performance_csv(perf_path, app_state.portfolio)
        repo.add_artifact(run_id, "performance_csv", str(perf_path))
        
        ret_path = base_path / "monthly_returns.csv"
        export_monthly_returns_csv(ret_path, app_state.portfolio)
        repo.add_artifact(run_id, "monthly_returns_csv", str(ret_path))

        performance = pd.read_csv(perf_path)
        performance_records = performance.to_dict(orient="records")

        rolling = compute_rolling_metrics(performance, risk_free_series=app_state.risk_free.series)
        rolling_path = base_path / "rolling_metrics.csv"
        export_rolling_metrics_csv(rolling_path, rolling)
        repo.add_artifact(run_id, "rolling_metrics_csv", str(rolling_path))
        if not rolling.empty:
            rolling_records = rolling.to_dict(orient="records")

    calendar = build_canonical_calendar(
        app_state.prices,
        benchmark_prices=app_state.benchmark_prices,
        total_values=app_state.portfolio.daily_values,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "RUN_STATE prices_rows=%s prices_cols=%s holdings_rows=%s holdings_cols=%s",
        len(app_state.prices),
        list(app_state.prices.columns) if not app_state.prices.empty else [],
        len(app_state.portfolio.holdings_daily),
        list(app_state.portfolio.holdings_daily.columns) if not app_state.portfolio.holdings_daily.empty else [],
    )

    attribution = compute_attribution(
        app_state.prices,
        app_state.portfolio.holdings_daily,
        app_state.portfolio.daily_values,
        app_state.portfolio.daily_returns,
        calendar=calendar,
    )
    try:
        from src.utils_memory import log_rss
        log_rss("after_attribution")
    except Exception:
        pass
    attribution_summary_path = base_path / "attribution_summary.json"
    export_attribution_summary_json(attribution_summary_path, attribution.summary)
    repo.add_artifact(run_id, "attribution_summary_json", str(attribution_summary_path))

    attribution_ts_path = base_path / "attribution_timeseries.csv"
    export_attribution_timeseries_csv(attribution_ts_path, attribution.timeseries)
    repo.add_artifact(run_id, "attribution_timeseries_csv", str(attribution_ts_path))

    weights = pd.Series(dtype=float)
    cash_weight = 0.0
    tickers = (
        sorted(app_state.prices["ticker"].dropna().unique().tolist())
        if not app_state.prices.empty and "ticker" in app_state.prices.columns
        else []
    )

    if tickers and not app_state.portfolio.holdings_daily.empty and not app_state.portfolio.daily_values.empty:
        latest_holdings = app_state.portfolio.holdings_daily
        latest_dates = pd.to_datetime(latest_holdings["date"], errors="coerce").dt.normalize()
        latest_date = latest_dates.max()
        latest = latest_holdings[latest_dates == latest_date]

        price_dates = pd.to_datetime(app_state.prices["date"], errors="coerce").dt.normalize()
        prices_latest = app_state.prices[price_dates == latest_date]
        if not prices_latest.empty:
            price_series = prices_latest.set_index("ticker")["adj_close"]
            values = latest.set_index("ticker")["quantity"].mul(price_series, fill_value=0.0)
            total_value = app_state.portfolio.daily_values["value"].iloc[-1]
            weights = values / total_value if total_value else values * 0.0
            cash_weight = float(max(0.0, 1.0 - weights.sum()))

    weights = weights.reindex(tickers).fillna(0.0) if tickers else weights

    assert_metrics_registered(["factor_tilts"])
    factor_output = compute_factor_tilts(app_state.scores, weights)
    factor_tilts_path = base_path / "factor_tilts.json"
    export_factor_tilts_json(factor_tilts_path, factor_output.summary, factor_output.details)
    repo.add_artifact(run_id, "factor_tilts_json", str(factor_tilts_path))

    cov = None
    tail_returns = None
    n_obs = 0
    var_alpha = 0.05
    var_value = 0.0
    if tickers and not calendar.empty:
        online = OnlineCovariance(len(tickers))
        portfolio_returns = app_state.portfolio.daily_returns.copy()
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index, errors="coerce").normalize()
        aligned_returns = portfolio_returns.reindex(calendar).fillna(0.0)
        pr_vals = aligned_returns.to_numpy(dtype=np.float64)
        var_value = float(np.quantile(pr_vals, var_alpha)) if pr_vals.size else 0.0
        best_diff = None

        for idx, (_, _, returns) in enumerate(iter_price_state(app_state.prices, tickers, calendar)):
            online.update(returns)
            n_obs += 1
            diff = abs(pr_vals[idx] - var_value) if idx < len(pr_vals) else None
            if diff is not None and (best_diff is None or diff < best_diff):
                best_diff = diff
                tail_returns = returns.copy()

        cov = online.covariance()

    risk_output = compute_risk_contributions_from_cov(
        cov,
        weights,
        tail_returns,
        cash_weight=cash_weight,
        var_alpha=var_alpha,
        var_value=var_value,
    )
    risk_csv_path = base_path / "risk_contribution.csv"
    export_risk_contribution_csv(risk_csv_path, risk_output.contributions)
    repo.add_artifact(run_id, "risk_contribution_csv", str(risk_csv_path))

    risk_json_path = base_path / "risk_contribution.json"
    export_risk_contribution_json(risk_json_path, risk_output.summary, risk_output.contributions)
    repo.add_artifact(run_id, "risk_contribution_json", str(risk_json_path))

    correlation_payload = compute_correlation_matrix_from_cov(
        cov,
        tickers,
        n_obs,
    )
    correlation_path = base_path / "correlation_matrix.json"
    export_correlation_matrix_json(correlation_path, correlation_payload)
    repo.add_artifact(run_id, "correlation_matrix_json", str(correlation_path))

    assert_metrics_registered(["hhi_concentration"])
    concentration_summary = build_hhi_summary(weights)
    concentration_path = base_path / "concentration_summary.json"
    export_concentration_summary_json(concentration_path, concentration_summary)
    repo.add_artifact(run_id, "concentration_summary_json", str(concentration_path))

    try:
        from src.utils_memory import log_rss
        log_rss("after_risk_correlation")
    except Exception:
        pass
    try:
        import gc
        del cov
        del tail_returns
        gc.collect()
    except Exception:
        pass

    if manifest.coverage_summary and "metric_status" in manifest.coverage_summary:
        status = correlation_payload.get("status", "unavailable")
        manifest.coverage_summary["metric_status"]["correlation_matrix"] = (
            "sufficient" if status in ("sufficient", "partial") else "insufficient"
        )
        reasons = correlation_payload.get("reasons", [])
        manifest.coverage_summary["metric_reasons"]["correlation_matrix"] = reasons
        export_coverage_summary_json(coverage_path, manifest.coverage_summary)
        repo.add_artifact(run_id, "coverage_summary_json", str(coverage_path))

    dates = pd.to_datetime(app_state.portfolio.daily_values.index, errors="coerce")
    cpi_result = get_cached_series("CPIAUCSL", allow_refresh=False)
    fed_result = get_cached_series("DFF", allow_refresh=False)
    vix_result = get_cached_series("VIXCLS", allow_refresh=False)
    cache_status = {
        "CPIAUCSL": cpi_result.status,
        "DFF": fed_result.status,
        "VIXCLS": vix_result.status,
    }
    macro_payload = compute_macro_regime_payload(
        dates,
        cpi_result.frame,
        fed_result.frame,
        vix_result.frame,
        cache_status=cache_status,
    )
    macro_path = base_path / "macro_regime_flags.csv"
    export_macro_regime_flags_csv(macro_path, macro_payload.flags)
    repo.add_artifact(run_id, "macro_regime_flags_csv", str(macro_path))

    macro_context_path = base_path / "macro_context.json"
    export_macro_context_json(
        macro_context_path,
        {
            "status": macro_payload.status,
            "available_series": macro_payload.available_series,
            "missing_series": macro_payload.missing_series,
            "tags": macro_payload.tags,
            "warnings": macro_payload.warnings,
            "as_of": macro_payload.as_of,
            "cache_status": macro_payload.cache_status,
        },
    )
    repo.add_artifact(run_id, "macro_context_json", str(macro_context_path))

    macro_summary_path = base_path / "macro_regime_summary.json"
    export_macro_regime_summary_json(macro_summary_path, {
        "status": macro_payload.status,
        "missing_series": macro_payload.missing_series,
        "as_of": macro_payload.as_of,
    })
    repo.add_artifact(run_id, "macro_regime_summary_json", str(macro_summary_path))

    comparison = compute_benchmark_comparison(
        app_state.portfolio.daily_returns,
        app_state.portfolio.daily_values,
        app_state.benchmark_prices,
    )
    bench_summary_path = base_path / "benchmark_comparison.json"
    export_benchmark_comparison_json(bench_summary_path, comparison.summary)
    repo.add_artifact(run_id, "benchmark_comparison_json", str(bench_summary_path))

    bench_timeseries_path = base_path / "benchmark_timeseries.csv"
    export_benchmark_timeseries_csv(bench_timeseries_path, comparison.timeseries)
    repo.add_artifact(run_id, "benchmark_timeseries_csv", str(bench_timeseries_path))

    last_date = None
    if not app_state.portfolio.daily_values.empty:
        last_date = app_state.portfolio.daily_values.index.max()
        if hasattr(last_date, "strftime"):
            last_date = last_date.strftime("%Y-%m-%d")

    diagnostics = generate_diagnostics(
        summary={"last_date": last_date} if last_date else None,
        attribution_summary=attribution.summary,
        risk_contribution={
            "summary": risk_output.summary,
            "contributions": risk_output.contributions.to_dict(orient="records"),
        },
        benchmark_comparison=comparison.summary,
        benchmark_timeseries=comparison.timeseries.to_dict(orient="records") if not comparison.timeseries.empty else [],
        performance=performance_records,
        rolling_metrics=rolling_records,
        coverage_summary=manifest.coverage_summary,
        weights=weights,
    )
    diagnostics_path = base_path / "diagnostics.json"
    export_diagnostics_json(diagnostics_path, diagnostics_payload(run_id, diagnostics))
    repo.add_artifact(run_id, "diagnostics_json", str(diagnostics_path))
