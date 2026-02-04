"""
Market Data Service

Handles fetching, validation, and caching of market data for tickers.
Abstracts market data operations from the API layer.
"""

import logging
from typing import List
import pandas as pd
from market_data.store import MarketDataStore
from market_data.contracts import MarketDataError

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching and validating market data."""
    
    def __init__(self, store: MarketDataStore | None = None):
        """
        Initialize the market data service.
        
        Args:
            store: MarketDataStore instance (defaults to singleton)
        """
        self.store = store or MarketDataStore.default()
    
    def fetch_batch(
        self,
        tickers: List[str],
        trade_dates: List[str],
    ) -> tuple[List[str], List[str]]:
        """
        Fetch market data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            trade_dates: List of trade dates to ensure coverage for
            
        Returns:
            Tuple of (successful_tickers, failed_tickers)
            
        Raises:
            MarketDataError: If a ticker fails with a known error code
        """
        import gc
        import resource
        
        failed_tickers: List[str] = []
        successful_tickers: List[str] = []
        
        if not tickers:
            return successful_tickers, failed_tickers
        
        logger.info("TICKER_FETCH_START total_tickers=%s", len(tickers))
        logger.info(
            "TICKER_FETCH_DATES min=%s max=%s count=%s",
            min(trade_dates),
            max(trade_dates),
            len(trade_dates)
        )
        
        for i, ticker in enumerate(tickers):
            logger.info("TICKER_FETCH_BEGIN ticker=%s progress=%s/%s", ticker, i + 1, len(tickers))
            
            try:
                prices = self.store.get_prices(
                    ticker,
                    start=str(min(trade_dates)),
                    end=str(max(trade_dates)),
                )
                self.store.ensure_coverage(prices, trade_dates, ticker)
                logger.info("TICKER_FETCH_SUCCESS ticker=%s rows=%s", ticker, len(prices))
                successful_tickers.append(ticker)
                
                # Free memory immediately after processing
                del prices
                gc.collect()
                logger.info("TICKER_FETCH_CLEANUP_DONE ticker=%s", ticker)
                
            except MarketDataError as exc:
                # Re-raise known market data errors to be handled by caller
                logger.error(
                    "MARKET_DATA_ERROR ticker=%s error_code=%s message=%s",
                    ticker,
                    exc.error_code,
                    exc.message,
                    extra={"details": exc.details}
                )
                raise
                
            except Exception as exc:
                # Log unexpected errors but continue with other tickers
                logger.exception("TICKER_FETCH_EXCEPTION ticker=%s", ticker)
                logger.warning("TICKER_FETCH_FAILED ticker=%s error=%s", ticker, exc)
                failed_tickers.append(ticker)
            
            # Log memory usage periodically
            if (i + 1) % 5 == 0 or (i + 1) == len(tickers):
                try:
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                    logger.info(
                        "RUN_MEMORY ticker=%s progress=%s/%s rss_mb=%.1f",
                        ticker,
                        i + 1,
                        len(tickers),
                        rss_mb
                    )
                except Exception:
                    pass
        
        if failed_tickers:
            logger.warning("RUN_TICKERS_FAILED count=%s tickers=%s", len(failed_tickers), failed_tickers)
        
        return successful_tickers, failed_tickers
    
    @staticmethod
    def get_user_friendly_error_message(error_code: str, ticker: str) -> str:
        """
        Map MarketDataError codes to user-friendly messages.
        
        Args:
            error_code: The MarketDataError error code
            ticker: The ticker symbol
            
        Returns:
            User-friendly error message
        """
        messages = {
            "MARKET_DATA_FETCH_EMPTY": f"No price data available for {ticker}. Please verify the ticker symbol is correct and has trading history.",
            "MARKET_DATA_MALFORMED": f"Unable to process market data for {ticker}. This may be a temporary issue with the data provider. Please try again later.",
            "MARKET_DATA_STALE": f"Market data for {ticker} is outdated and doesn't cover the required date range. This may indicate a data provider issue.",
            "MARKET_DATA_MISSING_DATE": f"Market data for {ticker} is incomplete (missing required date information). This ticker may not be fully supported.",
            "MARKET_DATA_FETCH_FAILED": f"Failed to fetch market data for {ticker}. Please check your internet connection and try again.",
        }
        return messages.get(error_code, f"Error fetching market data for {ticker}.")
