from __future__ import annotations

from datetime import date

import pandas as pd

from src.portfolio import compute_irr


def test_single_root_ok() -> None:
    flows = pd.Series([-10000.0], index=[date(2023, 1, 1)])
    out = compute_irr(flows, terminal_value=12000.0, valuation_end_date=date(2023, 12, 31))
    assert out.status == "ok"
    assert out.value is not None
    assert 0.18 < out.value < 0.22


def test_no_root_returned() -> None:
    flows = pd.Series([1000.0, 2000.0], index=[date(2023, 1, 1), date(2023, 6, 1)])
    out = compute_irr(flows, terminal_value=3000.0, valuation_end_date=date(2023, 12, 31))
    assert out.status == "no_root"
    assert out.value is None


def test_multi_root_flagged() -> None:
    flows = pd.Series(
        [-1000.0, 5000.0, -4500.0],
        index=[date(2023, 1, 1), date(2023, 5, 1), date(2023, 9, 1)],
    )
    out = compute_irr(flows, terminal_value=800.0, valuation_end_date=date(2023, 12, 31))
    assert out.status == "ambiguous_multi_root"
    assert out.value is None


def test_near_total_loss_converges() -> None:
    flows = pd.Series([-10000.0], index=[date(2023, 1, 1)])
    out = compute_irr(flows, terminal_value=110.0, valuation_end_date=date(2023, 12, 31))
    assert out.status == "ok"
    assert out.value is not None
    assert -0.995 < out.value < -0.97
