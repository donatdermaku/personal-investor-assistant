"""
Full Pipeline Analysis Script
Runs a comprehensive CSV through the entire pipeline, measuring timing at each stage.
Mirrors EXACTLY what the POST /run endpoint does.
"""
import sys
import os
import io
import time
import gc
import json
import uuid
import resource
import logging
import traceback

import pandas as pd

# Setup logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline_test")

# ── Helpers ──────────────────────────────────────────────────────

def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def timed_section(name):
    class _Timer:
        def __init__(self):
            self.elapsed = 0
        def __enter__(self):
            self._start = time.perf_counter()
            self._rss_start = rss_mb()
            logger.info("▶ START: %s (RSS=%.1f MB)", name, self._rss_start)
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._start
            rss_end = rss_mb()
            logger.info(
                "◀ END: %s — %.2fs (RSS %.1f→%.1f MB, Δ%.1f MB)",
                name, self.elapsed, self._rss_start, rss_end, rss_end - self._rss_start,
            )
    return _Timer()


# ── Main ─────────────────────────────────────────────────────

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_comprehensive.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PIPELINE ANALYSIS — %s", csv_path)
    logger.info("=" * 70)

    timings = {}
    overall_start = time.perf_counter()

    # ── 1. Read CSV ────────────────────────────────────────────
    with timed_section("1. Read CSV") as t:
        df = pd.read_csv(csv_path)
        logger.info("   Rows: %d, Columns: %s", len(df), list(df.columns))
    timings["1_read_csv"] = t.elapsed

    # ── 2. Validate & Prepare Ledger ───────────────────────────
    with timed_section("2. Validate & Prepare Ledger") as t:
        from src.services.portfolio_service import PortfolioService
        portfolio_service = PortfolioService()
        validated, errors = portfolio_service.validate_and_prepare_ledger(df)
        if errors:
            logger.error("   VALIDATION ERRORS: %s", errors)
            sys.exit(1)
        logger.info("   Validated rows: %d, columns: %s", len(validated), list(validated.columns))
    timings["2_validate_ledger"] = t.elapsed

    # ── 3. Extract Tickers ─────────────────────────────────────
    with timed_section("3. Extract Tickers") as t:
        tickers = portfolio_service.extract_tickers(validated)
        logger.info("   Tickers (%d): %s", len(tickers), tickers)
    timings["3_extract_tickers"] = t.elapsed

    # ── 4. Compute Trade Dates ─────────────────────────────────
    with timed_section("4. Compute Trade Dates") as t:
        trade_dates = pd.to_datetime(validated["date"], errors="coerce").dt.date.dropna().unique().tolist()
        logger.info("   Trade dates: %d (min=%s, max=%s)", len(trade_dates), min(trade_dates), max(trade_dates))
    timings["4_trade_dates"] = t.elapsed

    # ── 5. Market Data Fetch (MarketDataService.fetch_batch) ───
    with timed_section("5. MarketDataService.fetch_batch") as t:
        from src.services.market_data_service import MarketDataService
        from market_data.contracts import MarketDataError
        market_data_service = MarketDataService()
        failed_tickers = []
        try:
            successful, failed_tickers = market_data_service.fetch_batch(tickers, trade_dates)
            logger.info("   Successful: %s, Failed: %s", successful, failed_tickers)
        except MarketDataError as exc:
            logger.error("   MarketDataError: %s %s", exc.error_code, exc.message)
            logger.error("   Details: %s", exc.details)
        except Exception as exc:
            logger.error("   UNEXPECTED ERROR: %s", exc)
            traceback.print_exc()
    timings["5_market_data_fetch"] = t.elapsed

    if failed_tickers:
        logger.warning("⚠ Failed tickers: %s", failed_tickers)

    gc.collect()

    # ── 6. Save Portfolio Inputs ───────────────────────────────
    with timed_section("6. Save Portfolio Inputs") as t:
        from storage.datamanager import data_manager
        user_id = data_manager.get_current_user_id()
        portfolio_id = data_manager.get_main_portfolio_id(user_id)
        data_manager.save_portfolio_inputs(portfolio_id, validated, None)
        logger.info("   Portfolio ID: %d", portfolio_id)
    timings["6_save_inputs"] = t.elapsed

    # ── 7. compute_app_state (Full Pipeline) ───────────────────
    run_id = str(uuid.uuid4())
    with timed_section("7. compute_app_state (FULL PIPELINE)") as t:
        from src.pipeline import compute_app_state
        try:
            app_state = compute_app_state(
                portfolio_id=portfolio_id,
                run_id=run_id,
                save_run=True,
                source_override="Ledger",
                uploads_active=True,
                run_type="uploaded",
                trade_tickers=tickers,
            )
            logger.info("   ✓ compute_app_state SUCCESS")
            manifest = app_state.run_manifest
            logger.info("   Run ID: %s", manifest.run_id if manifest else "N/A")
        except Exception as exc:
            logger.error("   ✗ compute_app_state FAILED: %s", exc)
            traceback.print_exc()
            app_state = None
    timings["7_compute_app_state"] = t.elapsed

    # ── 8. save_artifacts ──────────────────────────────────────
    if app_state:
        with timed_section("8. save_artifacts") as t:
            from src.pipeline import save_artifacts
            try:
                save_artifacts(app_state)
                logger.info("   ✓ Artifacts saved")
            except Exception as exc:
                logger.error("   ✗ save_artifacts FAILED: %s", exc)
                traceback.print_exc()
        timings["8_save_artifacts"] = t.elapsed

        # ── 9. Verify Artifacts ────────────────────────────────
        with timed_section("9. Verify Artifacts") as t:
            from src.utils_io import ROOT
            exports_dir = ROOT / "data" / "exports" / run_id
            expected_artifacts = [
                "summary.json",
                "performance.csv",
                "monthly_returns.csv",
                "attribution_summary.json",
                "attribution_timeseries.csv",
                "risk_contribution.json",
                "rolling_metrics.csv",
                "macro_regime_flags.csv",
                "macro_regime_summary.json",
                "macro_context.json",
                "benchmark_comparison.json",
                "benchmark_timeseries.csv",
                "concentration_summary.json",
                "factor_tilts.json",
                "diagnostics.json",
                "correlation_matrix.json",
                "coverage_summary.json",
                "risk_free_series.csv",
                "corporate_actions_events.csv",
                "data_contracts.json",
                "report.html",
            ]
            found = []
            missing_artifacts = []
            for name in expected_artifacts:
                path = exports_dir / name
                if path.exists():
                    size = path.stat().st_size
                    found.append(f"{name} ({size:,} bytes)")
                else:
                    missing_artifacts.append(name)

            logger.info("   Found artifacts (%d):", len(found))
            for a in found:
                logger.info("     ✓ %s", a)
            if missing_artifacts:
                logger.warning("   Missing artifacts (%d):", len(missing_artifacts))
                for a in missing_artifacts:
                    logger.warning("     ✗ %s", a)
        timings["9_verify_artifacts"] = t.elapsed

        # ── 10. Inspect summary.json ───────────────────────────
        with timed_section("10. Inspect Key Outputs") as t:
            summary_path = exports_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                logger.info("   SUMMARY:")
                logger.info("     source: %s", summary.get("source"))
                logger.info("     TWR: %s", summary.get("twr"))
                logger.info("     MWR: %s", summary.get("mwr"))
                logger.info("     final_value: %s", summary.get("final_value"))
                logger.info("     last_date: %s", summary.get("last_date"))
                logger.info("     max_drawdown: %s", summary.get("max_drawdown"))
                logger.info("     errors: %s", summary.get("errors"))

            perf_path = exports_dir / "performance.csv"
            if perf_path.exists():
                perf_df = pd.read_csv(perf_path)
                logger.info("   PERFORMANCE: %d rows, date range: %s → %s",
                    len(perf_df), perf_df["date"].min() if "date" in perf_df.columns else "?",
                    perf_df["date"].max() if "date" in perf_df.columns else "?",
                )

            coverage_path = exports_dir / "coverage_summary.json"
            if coverage_path.exists():
                cov = json.loads(coverage_path.read_text())
                logger.info("   COVERAGE: status=%s, score=%s",
                    cov.get("status"), cov.get("score"),
                )

            diag_path = exports_dir / "diagnostics.json"
            if diag_path.exists():
                diag = json.loads(diag_path.read_text())
                diag_items = diag.get("diagnostics", [])
                logger.info("   DIAGNOSTICS: %d items", len(diag_items))
                for d in diag_items[:5]:
                    logger.info("     - [%s] %s", d.get("level"), d.get("message", "")[:100])
        timings["10_inspect_outputs"] = t.elapsed

    # ── Final Summary ──────────────────────────────────────────
    overall_elapsed = time.perf_counter() - overall_start
    timings["TOTAL"] = overall_elapsed

    logger.info("")
    logger.info("=" * 70)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 70)
    for key, val in timings.items():
        pct = (val / overall_elapsed) * 100 if overall_elapsed > 0 else 0
        bar = "█" * int(pct / 2)
        logger.info("  %-40s %8.2fs  %5.1f%%  %s", key, val, pct, bar)
    logger.info("  %-40s %8.2fs", "TOTAL", overall_elapsed)
    logger.info("  Peak RSS: %.1f MB", rss_mb())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
