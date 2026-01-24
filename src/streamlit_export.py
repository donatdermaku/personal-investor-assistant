from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from typing import Iterable

import pandas as pd

from src.portfolio import PortfolioResult, compute_drawdown, compute_monthly_returns


def export_summary_html(path: Path, sections: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join([f"<h2>{title}</h2><p>{content}</p>" for title, content in sections])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Summary</title></head><body>{body}</body></html>"
    path.write_text(html, encoding="utf-8")



def export_summary_json(path: Path, portfolio: PortfolioResult, manifest=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base payload
    payload = {
        "source": portfolio.source,
        "twr": portfolio.twr,
        "mwr": portfolio.mwr,
        "errors": portfolio.errors,
    }
    
    # Add manifest details if available
    if manifest:
        payload.update({
            "run_id": manifest.run_id,
            "input_hash": manifest.input_hash,
            "data_hash": manifest.data_hash,
            "timestamp": manifest.timestamp,
        })

    if portfolio.daily_values.empty:
        payload.update({
            "final_value": None,
            "last_date": None,
            "max_drawdown": None,
        })
        path.write_text(pd.Series(payload).to_json(), encoding="utf-8")
        return

    values = portfolio.daily_values.copy()
    max_drawdown = compute_drawdown(values["value"]).min()
    
    payload.update({
        "final_value": float(values["value"].iloc[-1]),
        "last_date": values.index[-1].strftime("%Y-%m-%d"),
        "max_drawdown": float(max_drawdown),
    })
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


def export_attribution_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def export_attribution_timeseries_csv(path: Path, timeseries: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if timeseries.empty:
        pd.DataFrame(columns=["date", "allocation", "selection", "interaction", "total_return"]).to_csv(path, index=False)
        return
    timeseries.to_csv(path, index=False)


def export_risk_contribution_csv(path: Path, risk_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if risk_df.empty:
        pd.DataFrame(
            columns=[
                "ticker",
                "volatility_contribution",
                "volatility_pct",
                "var_contribution",
                "var_pct",
            ]
        ).to_csv(path, index=False)
        return
    risk_df.to_csv(path, index=False)


def export_risk_contribution_json(path: Path, summary: dict, risk_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "contributions": risk_df.to_dict(orient="records") if not risk_df.empty else [],
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def export_macro_regime_flags_csv(path: Path, flags: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if flags.empty:
        pd.DataFrame(
            columns=[
                "date",
                "inflation_yoy",
                "fed_funds",
                "vix",
                "rates_change_6m",
                "high_inflation",
                "rising_rates",
                "risk_off",
            ]
        ).to_csv(path, index=False)
        return
    out = flags.copy()
    for col in ["high_inflation", "rising_rates", "risk_off"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    out.to_csv(path, index=False)


def export_macro_regime_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def export_rolling_metrics_csv(path: Path, rolling: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rolling.empty:
        pd.DataFrame(
            columns=["date", "rolling_volatility", "rolling_sharpe", "rolling_drawdown"]
        ).to_csv(path, index=False)
        return
    rolling.to_csv(path, index=False)


def export_benchmark_comparison_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def export_benchmark_timeseries_csv(path: Path, timeseries: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if timeseries.empty:
        pd.DataFrame(
            columns=["date", "portfolio_return", "benchmark_return", "active_return", "relative_drawdown"]
        ).to_csv(path, index=False)
        return
    timeseries.to_csv(path, index=False)


def export_coverage_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def generate_html_report(app_state) -> str:
    """
    Generate a standalone HTML report from the AppState.
    """
    import datetime
    from src.glossary import GLOSSARY
    
    # Unpack state
    manifest = app_state.run_manifest
    port = app_state.portfolio
    
    run_id = manifest.run_id if manifest else "N/A"
    run_date = manifest.timestamp if manifest else datetime.datetime.now().isoformat()
    
    # Metrics
    twr = f"{port.twr:.2%}" if port.twr is not None else "--"
    mwr = f"{port.mwr:.2%}" if port.mwr is not None else "--"
    errors = port.errors
    
    # HTML Template
    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Investor Report - {run_date[:10]}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; color: #1f2937; line-height: 1.5; }}
            h1, h2, h3 {{ color: #111827; }}
            .header {{ padding-bottom: 2rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 2rem; }}
            .meta {{ font-size: 0.875rem; color: #6b7280; font-family: monospace; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
            .card {{ background: #f9fafb; padding: 1.5rem; border-radius: 0.5rem; }}
            .metric-label {{ font-size: 0.875rem; color: #6b7280; display: block; margin-bottom: 0.5rem; }}
            .metric-value {{ font-size: 1.5rem; font-weight: 600; }}
            .error {{ background: #fee2e2; color: #991b1b; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem; }}
            th, td {{ text-align: left; padding: 0.75rem; border-bottom: 1px solid #e5e7eb; }}
            th {{ font-weight: 600; color: #374151; }}
            .footer {{ margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #e5e7eb; color: #9ca3af; font-size: 0.875rem; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Personal Investor Report</h1>
            <div class="meta">Run ID: {run_id}</div>
            <div class="meta">Generated: {run_date}</div>
        </div>
    """
    
    if errors:
        html += '<div class="error"><strong>⚠️ Portfolio Errors detected:</strong><ul>'
        for e in errors:
            html += f"<li>{e}</li>"
        html += "</ul></div>"
        
    html += f"""
        <h2>Executive Summary</h2>
        <div class="grid">
            <div class="card">
                <span class="metric-label">Strategy Return (TWR)</span>
                <span class="metric-value">{twr}</span>
            </div>
            <div class="card">
                <span class="metric-label">Personal Return (MWR)</span>
                <span class="metric-value">{mwr}</span>
            </div>
            <div class="card">
                <span class="metric-label">Price Coverage</span>
                <span class="metric-value">{app_state.price_meta.covered}/{app_state.price_meta.total}</span>
            </div>
        </div>
        
        <h2>Holdings</h2>
        <table>
            <thead>
                <tr><th>Ticker</th><th>Name</th><th>Allocation</th></tr>
            </thead>
            <tbody>
    """
    
    # Top Holdings (Simple placeholder logic, ideally we have logic in portfolio for weights)
    # Since we don't have explicit weights data easily accessible in AppState without processing, 
    # we will just list watch tickers for now or use app_state.watch_tickers
    
    for t in app_state.watch_tickers[:10]:
        html += f"<tr><td>{t}</td><td>--</td><td>--</td></tr>"
        
    html += """
            </tbody>
        </table>
        
        <h2>Glossary</h2>
        <ul>
    """
    
    for term, definition in GLOSSARY.items():
         html += f"<li><strong>{term}</strong>: {definition}</li>"
         
    html += """
        </ul>
        <div class="footer">
            Generated by Personal Investor Assistant. This report is for informational purposes only.
        </div>
    </body>
    </html>
    """
    return html


def save_html_report(path: Path, app_state) -> None:
    html = generate_html_report(app_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
