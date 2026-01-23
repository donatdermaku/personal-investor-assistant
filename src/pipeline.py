import uuid
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.app_state import AppState
from src.manifest import create_manifest, RunManifest, compute_input_hash
from src.portfolio import load_portfolio, PortfolioResult
from src.streamlit_data import (
    get_prices, get_scores, get_fundamentals, 
    merge_coverage, market_status, get_benchmark_prices, 
    get_universe, get_news
)
from storage.datamanager import data_manager
from storage import repo

# Define standard export paths relative to ROOT/data
EXPORTS_DIR = Path("data/exports")

def compute_app_state(
    portfolio_id: int | None = None,
    save_run: bool = True
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
    run_id = str(uuid.uuid4())
    if save_run:
        # We don't have hash yet, will update later
        repo.create_run(run_id, portfolio_id, None, None)
    
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
    portfolio_result = load_portfolio(prices, watch_tickers, source_override=None, uploads_active=True)
    
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
    
    manifest = create_manifest(
        run_id=run_id,
        input_hash=input_hash,
        config_hash="default", # Placeholder
        market_data_hash="duckdb_latest", # Placeholder
        portfolio_result=portfolio_result
    )

    # 9. Update Run in DB
    if save_run:
        repo.update_run_complete(run_id, manifest.to_json())
    
    # 10. Assemble AppState
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

def save_artifacts(app_state: AppState):
    """
    Generate and save export files (HTML, JSON, CSV).
    Record them in DB.
    """
    from src.streamlit_export import (
        export_summary_json, 
        export_performance_csv, 
        export_monthly_returns_csv, 
        save_html_report
    )
    
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

