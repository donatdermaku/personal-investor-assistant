from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.portfolio import compute_irr


def test_terminal_value_uses_end_date_not_last_flow() -> None:
    flows = pd.Series([-10000.0], index=[date(2023, 1, 1)])
    irr = compute_irr(flows, terminal_value=12000.0, valuation_end_date=date(2023, 12, 31))
    assert irr.status == "ok"
    assert irr.value is not None
    assert 0.18 < irr.value < 0.22


def test_dividends_excluded_from_mwr() -> None:
    flows = pd.Series(dtype=float)
    irr = compute_irr(flows, terminal_value=10500.0, valuation_end_date=date(2023, 12, 31))
    assert irr.status == "no_root"
    assert irr.value is None


def test_deposit_withdrawal_sign_convention() -> None:
    flows = pd.Series([-10000.0, 2000.0], index=[date(2023, 1, 1), date(2023, 6, 1)])
    irr = compute_irr(flows, terminal_value=9000.0, valuation_end_date=date(2023, 12, 31))
    assert irr.status == "ok"
    assert irr.value is not None


def test_missing_end_date_raises() -> None:
    with pytest.raises(ValueError):
        compute_irr(pd.Series([-10000.0], index=[date(2023, 1, 1)]), terminal_value=11000.0)
