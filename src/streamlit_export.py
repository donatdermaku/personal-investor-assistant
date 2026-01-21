from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.portfolio import PortfolioResult, compute_drawdown, compute_monthly_returns


def export_summary_html(path: Path, sections: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join([f"<h2>{title}</h2><p>{content}</p>" for title, content in sections])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Summary</title></head><body>{body}</body></html>"
    path.write_text(html, encoding="utf-8")


def export_summary_json(path: Path, portfolio: PortfolioResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if portfolio.daily_values.empty:
        payload = {
            "source": portfolio.source,
            "twr": portfolio.twr,
            "mwr": portfolio.mwr,
            "final_value": None,
            "last_date": None,
            "max_drawdown": None,
            "errors": portfolio.errors,
        }
        path.write_text(pd.Series(payload).to_json(), encoding="utf-8")
        return

    values = portfolio.daily_values.copy()
    max_drawdown = compute_drawdown(values["value"]).min()
    payload = {
        "source": portfolio.source,
        "twr": portfolio.twr,
        "mwr": portfolio.mwr,
        "final_value": float(values["value"].iloc[-1]),
        "last_date": values.index[-1].strftime("%Y-%m-%d"),
        "max_drawdown": float(max_drawdown),
        "errors": portfolio.errors,
    }
    path.write_text(pd.Series(payload).to_json(), encoding="utf-8")


def export_performance_csv(path: Path, portfolio: PortfolioResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if portfolio.daily_values.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    perf = portfolio.daily_values.copy()
    perf["daily_return"] = portfolio.daily_returns.reindex(perf.index).fillna(0.0)
    perf["drawdown"] = compute_drawdown(perf["value"])
    perf = perf.reset_index().rename(columns={"index": "date"})
    perf.to_csv(path, index=False)


def export_monthly_returns_csv(path: Path, portfolio: PortfolioResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if portfolio.daily_returns.empty:
        pd.DataFrame(columns=["date", "return"]).to_csv(path, index=False)
        return
    monthly = compute_monthly_returns(portfolio.daily_returns)
    out = monthly.reset_index()
    out.columns = ["date", "return"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
