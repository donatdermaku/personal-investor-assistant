"""
Portfolio Service

Handles portfolio operations including CSV validation, run creation,
and orchestration of portfolio computations.
"""

import logging
import uuid
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass

from src.portfolio import validate_ledger
from src.pipeline import compute_app_state, save_artifacts
from storage import datamanager as data_manager

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a portfolio run creation."""
    run_id: str
    status: str
    timestamp: str | None
    warnings: dict | None = None


class PortfolioService:
    """Service for portfolio operations."""
    
    def __init__(self, market_data_service=None):
        """
        Initialize the portfolio service.
        
        Args:
            market_data_service: MarketDataService instance
        """
        from src.services.market_data_service import MarketDataService
        self.market_data_service = market_data_service or MarketDataService()
    
    def validate_and_prepare_ledger(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and prepare ledger data from uploaded CSV.
        
        Args:
            df: Raw DataFrame from CSV upload
            
        Returns:
            Tuple of (validated_df, errors)
        """
        # Normalize column names
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        
        # Handle shares vs quantity column naming
        if "shares" in df.columns and "quantity" not in df.columns:
            df = df.rename(columns={"shares": "quantity"})
        
        # Validate ledger
        validated, errors = validate_ledger(df)
        
        if errors:
            return validated, errors
        
        # Fill missing quantity with 0
        if "quantity" in validated.columns:
            validated["quantity"] = validated["quantity"].fillna(0)
        
        # Calculate amount if missing
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
        
        return validated, []
    
    def extract_tickers(self, ledger: pd.DataFrame) -> List[str]:
        """
        Extract unique ticker symbols from ledger.
        
        Args:
            ledger: Validated ledger DataFrame
            
        Returns:
            Sorted list of unique tickers (excluding CASH)
        """
        tickers = sorted({
            t for t in ledger["ticker"].astype(str).str.upper().tolist()
            if t != "CASH"
        })
        return tickers
    
    def create_run(
        self,
        portfolio_id: int,
        ledger_df: pd.DataFrame,
        run_type: str = "uploaded",
        source_override: str = "Ledger"
    ) -> RunResult:
        """
        Create a new portfolio run from validated ledger data.
        
        Args:
            portfolio_id: Portfolio ID
            ledger_df: Validated ledger DataFrame
            run_type: Type of run ("uploaded" or "demo")
            source_override: Source label for the run
            
        Returns:
            RunResult with run_id, status, and optional warnings
        """
        run_id = str(uuid.uuid4())
        
        logger.info("RUN_INPUT rows=%s cols=%s", len(ledger_df), list(ledger_df.columns))
        
        # Save portfolio inputs
        data_manager.save_portfolio_inputs(portfolio_id, ledger_df, None)
        
        # Extract tickers
        tickers = self.extract_tickers(ledger_df)
        logger.info(
            "RUN_TICKERS count=%s tickers=%s",
            len(tickers),
            tickers[:10] if len(tickers) > 10 else tickers
        )
        
        # Fetch market data for all tickers
        trade_dates = pd.to_datetime(ledger_df["date"], errors="coerce").dt.date.dropna().unique().tolist()
        
        failed_tickers = []
        if tickers:
            _, failed_tickers = self.market_data_service.fetch_batch(tickers, trade_dates)
        
        # Compute portfolio state
        logger.info("RUN_COMPUTE_START run_id=%s portfolio_id=%s", run_id, portfolio_id)
        
        app_state = compute_app_state(
            portfolio_id=portfolio_id,
            run_id=run_id,
            save_run=True,
            source_override=source_override,
            uploads_active=True,
            run_type=run_type,
        )
        
        logger.info("RUN_COMPUTE_SUCCESS run_id=%s", run_id)
        
        # Save artifacts
        save_artifacts(app_state)
        logger.info("RUN_ARTIFACTS_SAVED run_id=%s", run_id)
        
        manifest = app_state.run_manifest
        
        # Build result with warnings if any tickers failed
        warnings = None
        if failed_tickers:
            warnings = {
                "failed_tickers": {
                    "count": len(failed_tickers),
                    "tickers": failed_tickers,
                    "message": f"{len(failed_tickers)} ticker(s) failed to load market data. Results may be incomplete.",
                }
            }
        
        return RunResult(
            run_id=manifest.run_id if manifest else run_id,
            status="completed",
            timestamp=manifest.timestamp if manifest else None,
            warnings=warnings,
        )
