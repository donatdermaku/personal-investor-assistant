from __future__ import annotations

import pandas as pd

from market_data.contracts import CoverageSummaryContract, PriceSeriesContract


def test_price_series_contract_missing_columns() -> None:
    contract = PriceSeriesContract()
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "close": [100.0]})
    errors = contract.validate_frame(df)
    assert any(err.field == "columns" for err in errors)


def test_coverage_summary_contract_missing_fields() -> None:
    errors = CoverageSummaryContract({"status": "unknown"})
    assert any(err.field == "as_of" for err in errors)
