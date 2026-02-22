from __future__ import annotations

import numpy as np
import pandas as pd

from src.compute.factors import _calc_fundamental_metrics, _calc_price_metrics


def _legacy_build_ttm_rollup(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("fiscal_end").copy()
    res = pd.DataFrame({"fiscal_end": g["fiscal_end"].values})
    rolling_cols = {
        "RevenueTTM": "Revenue",
        "NetIncomeTTM": "NetIncome",
        "OpCFTTM": "OperatingCF",
        "CapexTTM": "CapitalExpenditures",
        "GrossProfitTTM": "GrossProfit",
        "EBITDATTM": "EBITDA",
        "InterestExpenseTTM": "InterestExpense",
    }
    for out_col, src_col in rolling_cols.items():
        res[out_col] = g[src_col].rolling(4, min_periods=1).sum().values
    res["SharesDilutedTTM"] = g["SharesDiluted"].rolling(4, min_periods=1).mean().values
    passthrough = [
        "Debt",
        "CashAndEquivalents",
        "TotalAssets",
        "TotalLiabilities",
        "CurrentAssets",
        "CurrentLiabilities",
        "filed",
        "sic",
        "cik",
        "entity_name",
        "ticker",
    ]
    for col in passthrough:
        res[col] = g[col].values
    res["FCFTTM"] = res["OpCFTTM"] - res["CapexTTM"].fillna(0)
    return res


def test_ttm_window_sql_matches_legacy_partial_history() -> None:
    fnds = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 5,
            "fiscal_end": pd.to_datetime(
                [
                    "2023-03-31",
                    "2023-06-30",
                    "2023-09-30",
                    "2022-12-31",
                    "2023-03-31",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-12-31",
                ]
            ),
            "Revenue": [10, 20, 30, 5, 6, 7, 8, 9],
            "NetIncome": [1, 2, 3, 0.5, 0.6, 0.7, 0.8, 0.9],
            "SharesDiluted": [100, 100, 100, 50, 50, 50, 50, 50],
            "OperatingCF": [2, 3, 4, 1, 1, 1, 1, 1],
            "CapitalExpenditures": [1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2],
            "GrossProfit": [6, 12, 18, 2, 2.5, 3, 3.5, 4],
            "EBITDA": [3, 4, 5, 1, 1.2, 1.4, 1.6, 1.8],
            "InterestExpense": [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
            "Debt": [5] * 8,
            "CashAndEquivalents": [1] * 8,
            "TotalAssets": [20] * 8,
            "TotalLiabilities": [8] * 8,
            "CurrentAssets": [6] * 8,
            "CurrentLiabilities": [3] * 8,
            "filed": pd.to_datetime(["2023-04-30"] * 8),
            "sic": ["3571"] * 8,
            "cik": ["1"] * 8,
            "entity_name": ["Entity"] * 8,
        }
    )

    rollup_new, _ = _calc_fundamental_metrics(fnds)
    rollup_old = (
        fnds.groupby("ticker", group_keys=False)
        .apply(_legacy_build_ttm_rollup)
        .reset_index(drop=True)
        .sort_values(["ticker", "fiscal_end"])
        .reset_index(drop=True)
    )

    compare_cols = [
        "RevenueTTM",
        "NetIncomeTTM",
        "SharesDilutedTTM",
        "OpCFTTM",
        "CapexTTM",
        "GrossProfitTTM",
        "EBITDATTM",
        "InterestExpenseTTM",
        "FCFTTM",
    ]
    merged = rollup_new.sort_values(["ticker", "fiscal_end"]).reset_index(drop=True)
    for col in compare_cols:
        assert np.allclose(
            pd.to_numeric(merged[col], errors="coerce"),
            pd.to_numeric(rollup_old[col], errors="coerce"),
            equal_nan=True,
            atol=1e-12,
        )


def test_momentum_252_sql_matches_manual() -> None:
    dates = pd.date_range("2022-01-03", periods=260, freq="B")
    p1 = np.linspace(100.0, 150.0, len(dates))
    p2 = np.linspace(50.0, 90.0, len(dates))
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "AAA", "date": dates, "adj_close": p1}),
            pd.DataFrame({"ticker": "BBB", "date": dates, "adj_close": p2}),
        ],
        ignore_index=True,
    )

    out = _calc_price_metrics(prices)
    out = out.set_index("ticker")

    expected_aaa = (p1[-1] / p1[-253]) - 1
    expected_bbb = (p2[-1] / p2[-253]) - 1
    assert abs(float(out.loc["AAA", "Mom12m"]) - expected_aaa) < 1e-12
    assert abs(float(out.loc["BBB", "Mom12m"]) - expected_bbb) < 1e-12
