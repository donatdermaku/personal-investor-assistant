#!/usr/bin/env python3
"""
Cache warmup CLI for pre-populating market data cache.

Run as a Render one-off job, NOT in startCommand.

Usage:
    python scripts/warmup_cache.py                  # Warm default tickers
    python scripts/warmup_cache.py --force          # Force refresh even if cached
    python scripts/warmup_cache.py --dry-run        # Print what would be warmed
    python scripts/warmup_cache.py --tickers SPY,QQQ,AAPL  # Specific tickers
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Add project root to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data.store import MarketDataStore, FIXED_EARLIEST_DATE
from market_data.contracts import MarketDataError


# Default benchmarks and common tickers to warm
DEFAULT_TICKERS = [
    # Benchmarks (critical)
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "VTI",    # Total US Market
    "BND",    # Total Bond
    # Common large caps
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
]

# Critical tickers - script exits non-zero if these fail
CRITICAL_TICKERS = {"SPY", "QQQ"}


def warmup_ticker(
    store: MarketDataStore,
    ticker: str,
    end_date: str,
    *,
    force: bool = False,
) -> tuple[bool, Optional[str]]:
    """Warm a single ticker's cache.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # get_prices will use the rate limiter automatically
        df = store.get_prices(ticker, FIXED_EARLIEST_DATE, end_date)
        if df.empty:
            return False, "Empty result"
        return True, None
    except MarketDataError as exc:
        return False, f"{exc.error_code}: {exc.message}"
    except Exception as exc:
        return False, str(exc)


def run_warmup(
    tickers: list[str],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run warmup for all tickers.
    
    Returns:
        Summary dict with results
    """
    store = MarketDataStore.default()
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tickers": len(tickers),
        "warmed": [],
        "skipped": [],
        "failed": [],
        "critical_failed": [],
        "elapsed_seconds": 0,
    }
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Cache Warmup - {len(tickers)} tickers")
    print(f"End date: {end_date}")
    print(f"Force refresh: {force}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")
    
    for i, ticker in enumerate(tickers, 1):
        prefix = f"[{i}/{len(tickers)}]"
        
        if dry_run:
            print(f"{prefix} {ticker}: would warm (dry-run)")
            results["skipped"].append(ticker)
            continue
        
        print(f"{prefix} {ticker}: warming...", end=" ", flush=True)
        
        success, error = warmup_ticker(store, ticker, end_date, force=force)
        
        if success:
            print("✅")
            results["warmed"].append(ticker)
        else:
            is_critical = ticker in CRITICAL_TICKERS
            marker = "❌ CRITICAL" if is_critical else "⚠️"
            print(f"{marker} {error}")
            results["failed"].append({"ticker": ticker, "error": error})
            if is_critical:
                results["critical_failed"].append(ticker)
    
    results["elapsed_seconds"] = round(time.time() - start_time, 1)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Warmed:  {len(results['warmed'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Failed:  {len(results['failed'])}")
    print(f"  Elapsed: {results['elapsed_seconds']}s")
    
    if results["critical_failed"]:
        print(f"\n⛔ CRITICAL FAILURES: {results['critical_failed']}")
    
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-warm market data cache for common tickers."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to warm (default: built-in list)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache is fresh"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be warmed without actually fetching"
    )
    
    args = parser.parse_args()
    
    # Parse ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = DEFAULT_TICKERS
    
    # Run warmup
    results = run_warmup(tickers, force=args.force, dry_run=args.dry_run)
    
    # Exit with error if critical tickers failed
    if results["critical_failed"]:
        print("Exiting with error: critical tickers failed")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
