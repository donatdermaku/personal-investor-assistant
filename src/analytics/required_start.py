from datetime import date
import pandas as pd


def compute_required_start_per_ticker(
    holdings_daily: pd.DataFrame,
    default_start: date | None = None
) -> dict[str, date]:
    """
    Computes the required start date for each ticker based on when it first appeared 
    in the portfolio holdings (non-zero position).
    
    Args:
        holdings_daily: DataFrame with columns ["date", "ticker", "quantity"].
        default_start: Optional fallback date if no holdings exist or logic fails.
        
    Returns:
        dict[str, date]: Mapping of ticker -> earliest holding date.
    """
    if holdings_daily.empty:
        return {}

    required_starts: dict[str, date] = {}
    
    # Ensure date is date object
    df = holdings_daily.copy()
    if "date" in df.columns:
         df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    
    # Filter for non-zero quantity if quantity exists
    if "quantity" in df.columns:
        df = df[df["quantity"] != 0]

    if "ticker" not in df.columns or "date" not in df.columns:
        return {}

    # Group by ticker and find min date
    min_dates = df.groupby("ticker")["date"].min()
    
    for ticker, start_date in min_dates.items():
        if pd.notna(start_date):
            required_starts[str(ticker)] = start_date
            
    return required_starts
