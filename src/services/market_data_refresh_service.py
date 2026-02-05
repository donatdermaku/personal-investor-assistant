"""
Market Data Refresh Service

Handles automated market data refresh for all active portfolio tickers.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

from market_data.store import MarketDataStore
from market_data.contracts import MarketDataError
from storage.repo import use_supabase

logger = logging.getLogger(__name__)


class MarketDataRefreshService:
    """Service for refreshing market data for active portfolios."""
    
    def get_active_tickers(self) -> List[str]:
        """
        Get all unique tickers from active portfolios.
        
        Returns:
            List of unique ticker symbols
        """
        if use_supabase():
            from storage_supabase.db import session_scope
            from storage_supabase import models
        else:
            from storage.db import session_scope
            from storage import models
        
        with session_scope() as session:
            # Get all unique tickers from trades
            trades = session.query(models.Trade.ticker).distinct().all()
            tickers = [t.ticker for t in trades if t.ticker]
            
            # Deduplicate and sort
            all_tickers = sorted(set(tickers))
            
            # Filter out CASH
            all_tickers = [t for t in all_tickers if t.upper() != "CASH"]
            
            logger.info(f"Found {len(all_tickers)} active tickers", tickers=all_tickers)
            return all_tickers
    
    def refresh_market_data(
        self, 
        tickers: List[str], 
        days_back: int = 30,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch latest market data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            days_back: Number of days to fetch
            force_refresh: Clear cache before fetching
            
        Returns:
            Summary dict with success/failure counts
        """
        if not tickers:
            logger.warning("No tickers provided for refresh")
            return {
                "success_count": 0,
                "failure_count": 0,
                "tickers_processed": [],
                "errors": [],
            }
        
        store = MarketDataStore.default()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        success_count = 0
        failure_count = 0
        errors = []
        
        logger.info(
            f"Starting market data refresh",
            ticker_count=len(tickers),
            date_range=f"{start_date} to {end_date}",
            force_refresh=force_refresh
        )
        
        for ticker in tickers:
            try:
                # Clear cache if force_refresh
                if force_refresh:
                    from market_data.persistent_cache import clear_stale_cache
                    clear_stale_cache(source="yahoo", key=ticker)
                
                # Fetch prices
                prices = store.get_prices(ticker, start_date, end_date)
                
                if prices.empty:
                    logger.warning(f"No data returned for {ticker}")
                    failure_count += 1
                    errors.append({
                        "ticker": ticker,
                        "error": "No data returned"
                    })
                else:
                    success_count += 1
                    logger.info(f"Successfully refreshed {ticker}", rows=len(prices))
                    
            except MarketDataError as e:
                logger.error(f"Market data error for {ticker}: {e.message}", error_code=e.error_code)
                failure_count += 1
                errors.append({
                    "ticker": ticker,
                    "error": e.message,
                    "error_code": e.error_code
                })
            except Exception as e:
                logger.error(f"Unexpected error for {ticker}: {str(e)}")
                failure_count += 1
                errors.append({
                    "ticker": ticker,
                    "error": str(e)
                })
        
        summary = {
            "success_count": success_count,
            "failure_count": failure_count,
            "total_tickers": len(tickers),
            "tickers_processed": tickers,
            "errors": errors,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            "Market data refresh complete",
            success=success_count,
            failures=failure_count,
            total=len(tickers)
        )
        
        return summary
    
    def validate_refresh_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check refresh results and return validation summary.
        
        Args:
            results: Results from refresh_market_data()
            
        Returns:
            Validation summary with status and recommendations
        """
        total = results.get("total_tickers", 0)
        success = results.get("success_count", 0)
        failures = results.get("failure_count", 0)
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        status = "healthy"
        recommendations = []
        
        if success_rate < 80:
            status = "degraded"
            recommendations.append("High failure rate detected. Check Yahoo Finance availability.")
        
        if success_rate < 50:
            status = "critical"
            recommendations.append("Critical: Over 50% of tickers failed to refresh.")
        
        if failures > 0:
            error_types = {}
            for error in results.get("errors", []):
                error_code = error.get("error_code", "UNKNOWN")
                error_types[error_code] = error_types.get(error_code, 0) + 1
            
            recommendations.append(f"Error breakdown: {error_types}")
        
        return {
            "status": status,
            "success_rate": round(success_rate, 2),
            "recommendations": recommendations,
            "total_tickers": total,
            "success_count": success,
            "failure_count": failures
        }
