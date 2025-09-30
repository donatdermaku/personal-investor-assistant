import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils_stats import (
    calc_ev_to_ebitda,
    calc_fcf_yield,
    calc_roic,
    industry_zscores,
    piotroski_f_score,
    winsorize,
)


def test_piotroski_score_hits_all_components():
    dates = pd.to_datetime(["2022-12-31", "2023-03-31", "2023-06-30"])  # quarterly
    df = pd.DataFrame(
        {
            "fiscal_end": dates,
            "NetIncomeTTM": [40, 45, 52],
            "OpCFTTM": [42, 48, 55],
            "TotalAssets": [200, 205, 210],
            "Debt": [80, 78, 75],
            "CurrentAssets": [60, 65, 72],
            "CurrentLiabilities": [40, 39, 38],
            "GrossProfitTTM": [120, 126, 135],
            "RevenueTTM": [300, 310, 325],
            "SharesDilutedTTM": [50, 49, 48],
        }
    )
    scores = piotroski_f_score(df)
    assert int(scores.iloc[-1]) == 9
    assert int(scores.iloc[0]) == 3


def test_financial_ratio_helpers_match_expected_golden_values():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "golden_tickers.json"
    payload = json.loads(fixture_path.read_text())
    for entry in payload:
        fcfy = calc_fcf_yield(
            pd.Series([entry["fcf_ttm"]]),
            pd.Series([entry["price"]]),
            pd.Series([entry["shares"]]),
        ).iloc[0]
        ev_ebitda = calc_ev_to_ebitda(
            pd.Series([entry["price"]]),
            pd.Series([entry["shares"]]),
            pd.Series([entry["debt"]]),
            pd.Series([entry["cash"]]),
            pd.Series([entry["ebitda_ttm"]]),
        ).iloc[0]
        roic = calc_roic(
            pd.Series([entry["net_income_ttm"]]),
            pd.Series([entry["interest_expense_ttm"]]),
            pd.Series([entry["total_assets"]]),
            pd.Series([entry["current_liabilities"]]),
            pd.Series([entry["cash"]]),
            pd.Series([entry["debt"]]),
        ).iloc[0]

        exp = entry["expected"]
        assert np.isclose(fcfy, exp["fcf_yield"], rtol=1e-3)
        assert np.isclose(ev_ebitda, exp["ev_to_ebitda"], rtol=1e-3)
        assert np.isclose(roic, exp["roic"], rtol=1e-3)


def test_industry_zscores_handles_single_member_industry():
    values = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    industries = pd.Series(["Tech", "Tech", "Utilities"], index=["A", "B", "C"])
    zscores = industry_zscores(values, industries)
    # Tech should be standardised, Utilities singleton should map to zero
    assert np.isclose(zscores.loc["C"], 0.0)
    assert np.isclose(zscores.loc["A"], -1.0)
    assert np.isclose(zscores.loc["B"], 1.0)


def test_winsorize_clips_extremes():
    series = pd.Series([1, 2, 3, 100])
    clipped = winsorize(series, lower=0.0, upper=0.75)
    assert clipped.iloc[-1] == pytest.approx(series.quantile(0.75))
