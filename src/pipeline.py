import uuid
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.app_state import AppState
from src.manifest import create_manifest, RunManifest, compute_input_hash
from src.coverage import build_coverage_summary
from src.utils_io import ROOT
from src.portfolio import load_portfolio, PortfolioResult
from src.streamlit_data import (
    get_prices, get_scores, get_fundamentals, 
    merge_coverage, market_status, get_benchmark_prices, 
    get_universe, get_news
)
from storage.datamanager import data_manager
from storage import repo

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
        # We don't have hash yet, will update later
        repo.create_run(run_id, portfolio_id, None, None, run_type=run_type)
    
    # 4. Load User Data (Watchlist)
    watch_tickers = data_manager.load_watchlist() or []
    
    # 5. Load Market Data
    # For now, we load what we need for the watchlist
    prices, price_meta = get_prices(market_state, watch_tickers)
    scores, scores_meta = get_scores(watch_tickers)
    fund, fund_meta = get_fundamentals(watch_tickers)
    
    # 6. Load Portfolio
    # Here we use the underlying 'load_portfolio' which now uses DataManager
    # Note: load_portfolio calls DataManager internally for trades/snapshot.
    portfolio_result = load_portfolio(
        prices,
        watch_tickers,
        source_override=source_override,
        uploads_active=uploads_active,
    )
    
    # 7. Benchmark
    bench_ticker = "SPY" # Default, should come from settings
    bench_prices, bench_meta = get_benchmark_prices(bench_ticker)
    
    # 8. Create Manifest
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
    
    coverage_summary = build_coverage_summary(
        prices,
        required_tickers=watch_tickers,
        benchmark_ticker=bench_ticker,
        benchmark_prices=bench_prices,
        as_of=portfolio_result.daily_values.index.max().strftime("%Y-%m-%d") if not portfolio_result.daily_values.empty else None,
    )

    manifest = create_manifest(
        run_id=run_id,
        input_hash=input_hash,
        config_hash="default", # Placeholder
        market_data_hash="duckdb_latest", # Placeholder
        portfolio_result=portfolio_result,
        coverage_summary=coverage_summary,
    )

    # 9. Persist latest holdings snapshot (if available)
    _persist_snapshot_from_result(portfolio_id, portfolio_result)

    # 10. Update Run in DB
    if save_run:
        repo.update_run_complete(run_id, manifest.to_json(), run_type=run_type)
    
    # 11. Assemble AppState
    app_state = AppState(
        run_manifest=manifest,
        portfolio=portfolio_result,
        prices=prices,
        scores=scores,
        watch_tickers=watch_tickers,
        price_meta=price_meta,
        fundamentals_meta=fund_meta,
        scores_meta=scores_meta,
        benchmark_prices=bench_prices,
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
        export_rolling_metrics_csv,
        export_benchmark_comparison_json,
        export_benchmark_timeseries_csv,
    )
    from src.analytics.attribution import compute_attribution
    from src.analytics.risk import compute_risk_contributions
    from src.analytics.rolling import compute_rolling_metrics
    from src.analytics.comparative import compute_benchmark_comparison
    from src.analytics.macro import compute_macro_regime_payload, load_cached_fred_series
    
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
    
    # HTML Report
    html_path = base_path / "report.html"
    save_html_report(html_path, app_state)
    repo.add_artifact(run_id, "html_report", str(html_path))
    
    # CSVs
    if not app_state.portfolio.daily_values.empty:
        perf_path = base_path / "performance.csv"
        export_performance_csv(perf_path, app_state.portfolio)
        repo.add_artifact(run_id, "performance_csv", str(perf_path))
        
        ret_path = base_path / "monthly_returns.csv"
        export_monthly_returns_csv(ret_path, app_state.portfolio)
        repo.add_artifact(run_id, "monthly_returns_csv", str(ret_path))

        performance = pd.read_csv(perf_path)

        rolling = compute_rolling_metrics(performance)
        rolling_path = base_path / "rolling_metrics.csv"
        export_rolling_metrics_csv(rolling_path, rolling)
        repo.add_artifact(run_id, "rolling_metrics_csv", str(rolling_path))

    attribution = compute_attribution(
        app_state.prices,
        app_state.portfolio.holdings_daily,
        app_state.portfolio.daily_values,
        app_state.portfolio.daily_returns,
    )
    attribution_summary_path = base_path / "attribution_summary.json"
    export_attribution_summary_json(attribution_summary_path, attribution.summary)
    repo.add_artifact(run_id, "attribution_summary_json", str(attribution_summary_path))

    attribution_ts_path = base_path / "attribution_timeseries.csv"
    export_attribution_timeseries_csv(attribution_ts_path, attribution.timeseries)
    repo.add_artifact(run_id, "attribution_timeseries_csv", str(attribution_ts_path))

    risk_prices = app_state.prices.copy()
    returns = pd.DataFrame()
    weights = pd.Series(dtype=float)
    cash_weight = 0.0
    if not risk_prices.empty and {"date", "ticker", "adj_close"}.issubset(risk_prices.columns):
        risk_prices["date"] = pd.to_datetime(risk_prices["date"], errors="coerce")
        returns = (
            risk_prices.pivot_table(index="date", columns="ticker", values="adj_close")
            .sort_index()
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        latest_holdings = app_state.portfolio.holdings_daily
        if not latest_holdings.empty and not app_state.portfolio.daily_values.empty:
            latest_date = pd.to_datetime(latest_holdings["date"], errors="coerce").max()
            latest = latest_holdings[latest_holdings["date"] == latest_date]
            prices = risk_prices[risk_prices["date"] == latest_date].set_index("ticker")["adj_close"]
            values = latest.set_index("ticker")["quantity"].mul(prices, fill_value=0.0)
            total_value = app_state.portfolio.daily_values["value"].iloc[-1]
            weights = values / total_value if total_value else values * 0.0
            cash_weight = float(max(0.0, 1.0 - weights.sum()))

    risk_output = compute_risk_contributions(returns, weights, cash_weight=cash_weight)
    risk_csv_path = base_path / "risk_contribution.csv"
    export_risk_contribution_csv(risk_csv_path, risk_output.contributions)
    repo.add_artifact(run_id, "risk_contribution_csv", str(risk_csv_path))

    risk_json_path = base_path / "risk_contribution.json"
    export_risk_contribution_json(risk_json_path, risk_output.summary, risk_output.contributions)
    repo.add_artifact(run_id, "risk_contribution_json", str(risk_json_path))

    if not app_state.portfolio.daily_values.empty:
        dates = pd.to_datetime(app_state.portfolio.daily_values.index, errors="coerce")
        cpi = load_cached_fred_series("CPIAUCSL")
        fed_funds = load_cached_fred_series("DFF")
        vix = load_cached_fred_series("VIXCLS")
        macro_payload = compute_macro_regime_payload(dates, cpi, fed_funds, vix)
        macro_path = base_path / "macro_regime_flags.csv"
        export_macro_regime_flags_csv(macro_path, macro_payload.flags)
        repo.add_artifact(run_id, "macro_regime_flags_csv", str(macro_path))

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
