from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def qa_checks(df: pd.DataFrame, min_size: int) -> dict:
    warnings = []
    if df.empty:
        warnings.append("Scores table is empty.")
    if len(df) < min_size:
        warnings.append(f"Universe size {len(df)} below {min_size}; using robust z-scores.")
    if df["Composite"].isna().all() or np.isclose(df["Composite"].fillna(0), 0).all():
        warnings.append("Composite scores are all NaN/zero.")
    missing_fund = df["has_fundamentals"].fillna(False).mean() if not df.empty else 0
    if missing_fund < 0.8:
        warnings.append("More than 20% of universe missing fundamentals.")
    missing_prices = df["has_prices"].fillna(False).mean() if not df.empty else 0
    if missing_prices < 0.8:
        warnings.append("More than 20% of universe missing price history.")

    return {
        "warnings": warnings,
        "universe_size": len(df),
        "min_size": min_size,
        "use_robust": len(df) < min_size,
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
