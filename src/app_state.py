from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.manifest import RunManifest
from src.portfolio import PortfolioResult
# check if CoverageMeta can be imported from streamlit_data without cycle
# streamlit_data imports NOTHING from here.
from src.coverage_meta import CoverageMeta
from src.risk_free import RiskFreeSeries

@dataclass
class AppState:
    """Unified state object for exports and reporting."""
    run_manifest: RunManifest | None
    portfolio: PortfolioResult
    prices: pd.DataFrame
    scores: pd.DataFrame
    watch_tickers: list[str]
    price_meta: CoverageMeta
    fundamentals_meta: CoverageMeta
    scores_meta: CoverageMeta
    benchmark_prices: pd.DataFrame
    risk_free: RiskFreeSeries
    market_state: str

    @property
    def has_data(self) -> bool:
        return not self.portfolio.daily_values.empty
