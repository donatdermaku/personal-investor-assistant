from __future__ import annotations

import json
import pathlib
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.utils_io import ROOT, PARQ, load_yaml


def _latest(prefix: str) -> pathlib.Path | None:
    files = sorted(PARQ.glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def _all(prefix: str) -> list[pathlib.Path]:
    return sorted(PARQ.glob(f"{prefix}_*.parquet"))


def _safe_read_parquet(path: pathlib.Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover
        print(f"Warning: failed to read parquet {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def _normalize_weights(tickers: list[str], weights_cfg: dict[str, float]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    w = pd.Series(weights_cfg or {}, dtype=float)
    w = w.reindex(tickers).fillna(0.0)
    if w.sum() <= 0:
        w = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        w = w / w.sum()
    return w


def _load_watchlist() -> tuple[list[str], dict[str, str], dict[str, float]]:
    watch = load_yaml(ROOT / "watchlist.yml") or {}
    tickers = [str(t).upper() for t in watch.get("tickers", [])]
    notes = watch.get("notes", {}) or {}
    weights = watch.get("weights", {}) or {}
    return tickers, notes, weights


def _load_qa() -> dict:
    qa_dir = ROOT / "data" / "qa"
    if not qa_dir.exists():
        return {}
    files = sorted(qa_dir.glob("qa_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _portfolio_stats(prices: pd.DataFrame, weights: pd.Series) -> tuple[list[dict], float | None, float | None]:
    price_history = []
    portfolio_drawdown = np.nan
    portfolio_return = np.nan
    if not prices.empty and not weights.empty:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        wide = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
        wide = wide.ffill().dropna(how="all")
        if not wide.empty:
            aligned_weights = weights.reindex(wide.columns).fillna(0.0)
            if aligned_weights.sum() <= 0:
                aligned_weights[:] = 1.0 / len(aligned_weights)
            returns = wide.pct_change().fillna(0)
            portfolio_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
            portfolio_index = (1 + portfolio_returns).cumprod()
            price_history = [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                for idx, val in portfolio_index.items()
            ]
            if not portfolio_index.empty:
                start_of_year = portfolio_index[portfolio_index.index >= pd.Timestamp(datetime.now(tz=timezone.utc).year, 1, 1)]
                if not start_of_year.empty:
                    base = start_of_year.iloc[0]
                    if base != 0:
                        portfolio_return = float(start_of_year.iloc[-1] / base - 1.0)
                if portfolio_return != portfolio_return:
                    portfolio_return = float(portfolio_index.iloc[-1] - 1.0)
                peak = portfolio_index.cummax()
                drawdown = portfolio_index / peak - 1.0
                if not drawdown.empty:
                    portfolio_drawdown = float(drawdown.iloc[-1])
    return price_history, portfolio_return, portfolio_drawdown


def _build_composite_history(scores_paths: list[pathlib.Path], weights: pd.Series, tickers: list[str]) -> list[dict]:
    composite_history = []
    for path in scores_paths:
        stem = path.stem
        try:
            date_part = stem.split("_")[-1]
            snap_date = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        df = _safe_read_parquet(path)
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        df = df.set_index("ticker")
        composite = df["Composite"].reindex(weights.index).fillna(0.0)
        composite_history.append({
            "date": snap_date.strftime("%Y-%m-%d"),
            "value": float((composite * weights).sum()),
        })
    return composite_history


def _build_watch_rows(scores: pd.DataFrame, prior_scores: pd.DataFrame, notes: dict, weights: pd.Series) -> list[dict]:
    rows = []
    if scores.empty:
        return rows
    scores = scores.set_index("ticker")
    prior_scores = prior_scores.set_index("ticker") if not prior_scores.empty else pd.DataFrame()
    for t in weights.index:
        row = scores.loc[t] if t in scores.index else None
        prev = prior_scores.loc[t] if not prior_scores.empty and t in prior_scores.index else None
        composite_delta = None
        composite_delta_7d = None
        if row is not None:
            composite_delta = row.get("composite_pct_change_1d")
            composite_delta_7d = row.get("composite_pct_change_7d")
        data_quality = []
        if row is None or not row.get("has_prices", False):
            data_quality.append("missing_prices")
        if row is None or not row.get("has_fundamentals", False):
            data_quality.append("missing_fundamentals")
        if row is not None and pd.notna(row.get("fundamentals_staleness_days")):
            if row.get("fundamentals_staleness_days") > 180:
                data_quality.append("stale_fundamentals")
        filed_val = row.get("filed") if row is not None else None
        if isinstance(filed_val, pd.Timestamp):
            filed_val = filed_val.strftime("%Y-%m-%d")
        rows.append({
            "ticker": t,
            "note": notes.get(t, ""),
            "price": None if row is None else float(row.get("Price", np.nan)),
            "composite": None if row is None else float(row.get("Composite", np.nan)),
            "composite_pct": None if row is None else float(row.get("composite_pct", np.nan)),
            "value_pct": None if row is None else float(row.get("value_pct", np.nan)),
            "quality_pct": None if row is None else float(row.get("quality_pct", np.nan)),
            "momentum_pct": None if row is None else float(row.get("momentum_pct", np.nan)),
            "composite_delta": None if composite_delta is None else float(composite_delta),
            "composite_delta_7d": None if composite_delta_7d is None else float(composite_delta_7d),
            "value_z": None if row is None else float(row.get("ValueZ", np.nan)),
            "quality_z": None if row is None else float(row.get("QualityZ", np.nan)),
            "momentum": None if row is None else float(row.get("MomScore", np.nan)),
            "piotroski": None if row is None else float(row.get("PiotroskiF", np.nan)),
            "volatility": None if row is None else float(row.get("Volatility30d", np.nan)),
            "sharpe": None if row is None else float(row.get("Sharpe1y", np.nan)),
            "industry": None if row is None else row.get("industry"),
            "filed": filed_val,
            "filing_url": None if row is None else _filing_url(row),
            "data_quality": data_quality,
            "weight": float(weights.get(t, 0.0)),
        })
    return rows


def _filing_url(row: pd.Series) -> str | None:
    cik = row.get("cik")
    if not cik or pd.isna(cik):
        return None
    return f"https://www.sec.gov/edgar/browse/?CIK={str(cik).zfill(10)}&owner=exclude"


def _build_pulse(scores: pd.DataFrame, prior_scores: pd.DataFrame, cfg: dict) -> dict:
    alerts = cfg.get("alerts", {}) if cfg else {}
    lookback_days = int(alerts.get("filings_lookback_days", 7))
    drawdown_alert = float(alerts.get("drawdown_alert_pct", -20)) / 100
    vol_spike_ratio = float(alerts.get("vol_spike_ratio", 1.5))

    pulse = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "top_movers": [],
        "volatility_changes": [],
        "drawdown_alerts": [],
        "new_filings": [],
        "data_quality": [],
        "top_opportunities": [],
        "top_risks": [],
    }

    if scores.empty:
        return pulse

    scores = scores.copy()
    scores["composite_pct_change_1d"] = scores["composite_pct_change_1d"].fillna(0)

    movers = scores.sort_values("composite_pct_change_1d", ascending=False)
    pulse["top_movers"] = (
        movers[["ticker", "composite_pct", "composite_pct_change_1d"]]
        .head(5)
        .to_dict(orient="records")
    )

    if not prior_scores.empty:
        merged = scores.merge(
            prior_scores[["ticker", "Volatility30d"]],
            on="ticker",
            how="left",
            suffixes=("", "_prior"),
        )
        merged["vol_ratio"] = merged["Volatility30d"] / merged["Volatility30d_prior"].replace({0: np.nan})

        vol_changes = merged.sort_values("vol_ratio", ascending=False)
        pulse["volatility_changes"] = (
            vol_changes[["ticker", "Volatility30d", "Volatility30d_prior", "vol_ratio"]]
            .head(5)
            .to_dict(orient="records")
        )

        spike = vol_changes[vol_changes["vol_ratio"] >= vol_spike_ratio]
        for _, row in spike.head(5).iterrows():
            pulse["data_quality"].append(
                f"Volatility spike for {row['ticker']} (x{row['vol_ratio']:.2f})."
            )

    drawdowns = scores[scores["Drawdown1y"] <= drawdown_alert]
    pulse["drawdown_alerts"] = (
        drawdowns[["ticker", "Drawdown1y", "composite_pct"]]
        .sort_values("Drawdown1y")
        .head(5)
        .to_dict(orient="records")
    )

    # ---- FIX: timezone-safe filings filter ----
    recent_cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)

    filings = scores[scores["filed"].notna()].copy()
    if not filings.empty:
        filings["filed"] = pd.to_datetime(
            filings["filed"],
            errors="coerce",
            utc=True
        )
        filings = filings.dropna(subset=["filed"])

        recent = filings[filings["filed"] >= recent_cutoff]
        pulse["new_filings"] = (
            recent[["ticker", "filed"]]
            .sort_values("filed", ascending=False)
            .head(10)
            .assign(filed=lambda df: df["filed"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        )
    # ------------------------------------------

    missing = scores[scores["missing_key_fields"].fillna("") != ""]
    if not missing.empty:
        for _, row in missing.head(5).iterrows():
            pulse["data_quality"].append(
                f"{row['ticker']} missing: {row['missing_key_fields']}"
            )

    opportunities = scores[scores["has_prices"] & scores["has_fundamentals"]].sort_values(
        "composite_pct", ascending=False
    )
    for _, row in opportunities.head(3).iterrows():
        rationale = (
            f"Composite {row['composite_pct']:.1f}pct, "
            f"momentum {row['momentum_pct']:.1f}pct, "
            f"ROIC {row['ROIC']:.2f}."
        )
        pulse["top_opportunities"].append(
            {"ticker": row["ticker"], "rationale": rationale}
        )

    risks = scores.sort_values("composite_pct", ascending=True)
    for _, row in risks.head(3).iterrows():
        rationale = (
            f"Composite {row['composite_pct']:.1f}pct, "
            f"drawdown {row['Drawdown1y']:.1%}, "
            f"volatility {row['Volatility30d']:.2f}."
        )
        pulse["top_risks"].append(
            {"ticker": row["ticker"], "rationale": rationale}
        )

    return pulse


def _pulse_markdown(pulse: dict) -> str:
    lines = ["# Daily Pulse", "", f"Generated: {pulse.get('generated_at', '')}", ""]
    def _section(title: str, items: list, formatter):
        lines.append(f"## {title}")
        if not items:
            lines.append("- None")
            lines.append("")
            return
        for item in items:
            lines.append(f"- {formatter(item)}")
        lines.append("")

    _section("Top Movers", pulse.get("top_movers", []), lambda r: f"{r['ticker']} ({r['composite_pct_change_1d']:.1f} pct-pts)" )
    _section("Volatility Changes", pulse.get("volatility_changes", []), lambda r: f"{r['ticker']} (x{r.get('vol_ratio', 0):.2f})")
    _section("Drawdown Alerts", pulse.get("drawdown_alerts", []), lambda r: f"{r['ticker']} ({r['Drawdown1y']:.1%})")
    _section("New Filings", pulse.get("new_filings", []), lambda r: f"{r['ticker']} ({r['filed']})")
    _section("Data Quality", pulse.get("data_quality", []), lambda r: str(r))
    _section("Top Opportunities", pulse.get("top_opportunities", []), lambda r: f"{r['ticker']}: {r['rationale']}")
    _section("Top Risks", pulse.get("top_risks", []), lambda r: f"{r['ticker']}: {r['rationale']}")
    return "\n".join(lines).strip() + "\n"


def _write_ticker_pages(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    scores_paths: list[pathlib.Path],
    tickers: list[str],
    reports: pathlib.Path,
    cfg: dict,
    env: Environment,
) -> None:
    data_dir = reports / "data" / "ticker"
    data_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = reports / "ticker"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    max_history = int(cfg.get("report", {}).get("max_history_snapshots", 260))
    history_paths = scores_paths[-max_history:]

    score_history_frames = []
    for path in history_paths:
        stem = path.stem
        try:
            date_part = stem.split("_")[-1]
            snap_date = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        df = _safe_read_parquet(path)
        if df.empty:
            continue
        df = df[df["ticker"].isin(tickers)].copy()
        df["snap_date"] = snap_date.strftime("%Y-%m-%d")
        score_history_frames.append(df)

    score_history = pd.concat(score_history_frames, ignore_index=True) if score_history_frames else pd.DataFrame()

    prices = prices.copy()
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"])

    tpl = env.get_template("ticker.html.j2")

    history_days = int(cfg.get("report", {}).get("ticker_history_days", 365))
    for t in tickers:
        price_series = []
        if not prices.empty:
            series = prices[prices["ticker"] == t].sort_values("date").tail(history_days)
            price_series = [
                {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                for d, v in zip(series["date"], series["adj_close"])
            ]

        factor_series = []
        if not score_history.empty:
            series = score_history[score_history["ticker"] == t].sort_values("snap_date")
            factor_series = [
                {
                    "date": row["snap_date"],
                    "value": float(row.get("Composite", np.nan)),
                    "value_score": float(row.get("ValueScore", np.nan)),
                    "quality_score": float(row.get("QualityScore", np.nan)),
                    "momentum_score": float(row.get("MomScore", np.nan)),
                }
                for _, row in series.iterrows()
            ]

        fundamentals_series = []
        if not score_history.empty:
            series = score_history[score_history["ticker"] == t].sort_values("snap_date")
            fundamentals_series = [
                {
                    "date": row["snap_date"],
                    "revenue": float(row.get("RevenueGrowthYoY", np.nan)),
                    "margin": float(row.get("GrossMargin", np.nan)),
                    "fcf_margin": float(row.get("FCFMargin", np.nan)),
                }
                for _, row in series.iterrows()
            ]

        payload = {
            "ticker": t,
            "price": price_series,
            "factors": factor_series,
            "fundamentals": fundamentals_series,
        }
        data_path = data_dir / f"{t}.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")

        row = scores[scores["ticker"] == t].iloc[0] if not scores.empty and t in scores["ticker"].values else None
        base_path = ".."
        html = tpl.render(
            generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            ticker=t,
            row=row,
            data_path=f"{base_path}/data/ticker/{t}.json",
            asset_path=f"{base_path}/assets",
            base_path=base_path,
        )
        (ticker_dir / f"{t}.html").write_text(html, encoding="utf-8")


def main() -> None:
    cfg = load_yaml(ROOT / "config.yml") or {}
    tickers, notes, weights_cfg = _load_watchlist()
    weights = _normalize_weights(tickers, weights_cfg)

    scores_path = _latest("scores_daily")
    fundamentals_path = _latest("fundamentals_quarterly")
    prices_path = _latest("prices_daily")

    scores = _safe_read_parquet(scores_path)
    fundamentals = _safe_read_parquet(fundamentals_path)
    prices = _safe_read_parquet(prices_path)

    watch_scores = scores[scores["ticker"].isin(tickers)] if not scores.empty else scores
    watch_fundamentals = fundamentals[fundamentals["ticker"].isin(tickers)] if not fundamentals.empty else fundamentals
    watch_prices = prices[prices["ticker"].isin(tickers)] if not prices.empty else prices

    prior_scores = pd.DataFrame()
    history_paths = _all("scores_daily")
    if len(history_paths) >= 2:
        prior_scores = _safe_read_parquet(history_paths[-2])

    composite_history = _build_composite_history(history_paths, weights, tickers)
    price_history, portfolio_return, portfolio_drawdown = _portfolio_stats(watch_prices, weights)
    latest_portfolio_composite = None
    if not watch_scores.empty and not weights.empty:
        latest_comp = watch_scores.set_index("ticker")["Composite"].reindex(weights.index).fillna(0.0)
        latest_portfolio_composite = float((latest_comp * weights).sum())

    warnings = []
    qa = _load_qa()
    warnings.extend(qa.get("warnings", []))

    if not prices.empty:
        latest_price_date = pd.to_datetime(prices["date"]).max()
        if latest_price_date is not None:
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            latest_norm = pd.to_datetime(latest_price_date)
            if latest_norm.tzinfo is None:
                latest_norm = latest_norm.tz_localize("UTC")
            else:
                latest_norm = latest_norm.tz_convert("UTC")
            latest_norm = latest_norm.normalize()
            lag_days = (now_utc - latest_norm).days
            if lag_days > 3:
                warnings.append(f"Prices are {lag_days} days old.")
    if not fundamentals.empty:
        latest_fund_date = pd.to_datetime(fundamentals["fiscal_end"]).max()
        if latest_fund_date is not None:
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            fund_norm = pd.to_datetime(latest_fund_date)
            if fund_norm.tzinfo is None:
                fund_norm = fund_norm.tz_localize("UTC")
            else:
                fund_norm = fund_norm.tz_convert("UTC")
            fund_norm = fund_norm.normalize()
            lag_quarters = (now_utc - fund_norm).days / 90
            if lag_quarters > 2:
                warnings.append("Fundamentals stale (>2 quarters).")
    if not tickers:
        warnings.append("No tickers available in watchlist.")

    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        autoescape=select_autoescape(),
    )

    watch_rows = _build_watch_rows(watch_scores, prior_scores, notes, weights)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "assets").mkdir(parents=True, exist_ok=True)

    assets_src = ROOT / "assets"
    for asset in ["report.css", "report.js", "charts.js"]:
        shutil.copy(assets_src / asset, reports / "assets" / asset)

    tpl = env.get_template("report.html.j2")
    base_path = "."
    html = tpl.render(
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        watch_rows=watch_rows,
        composite_history=composite_history,
        price_history=price_history,
        portfolio_return=portfolio_return,
        portfolio_drawdown=portfolio_drawdown,
        warnings=warnings,
        latest_portfolio_composite=latest_portfolio_composite,
        asset_path=f"{base_path}/assets",
        base_path=base_path,
    )

    (reports / "latest.html").write_text(html, encoding="utf-8")
    (reports / "index.html").write_text(html, encoding="utf-8")

    ym = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    (reports / f"{ym}").mkdir(parents=True, exist_ok=True)
    (reports / f"{ym}/index.html").write_text(html, encoding="utf-8")

    _write_ticker_pages(scores, prices, history_paths, tickers, reports, cfg, env)

    pulse = _build_pulse(scores, prior_scores, cfg)
    pulse_json = json.dumps(pulse, indent=2)
    pulse_md = _pulse_markdown(pulse)

    pulse_tpl = env.get_template("pulse.html.j2")
    pulse_html = pulse_tpl.render(pulse=pulse, asset_path="./assets", base_path=".")

    (reports / "pulse").mkdir(parents=True, exist_ok=True)
    (reports / "pulse.json").write_text(pulse_json, encoding="utf-8")
    (reports / "pulse.md").write_text(pulse_md, encoding="utf-8")
    (reports / "pulse.html").write_text(pulse_html, encoding="utf-8")

    run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    archive_dir = reports / "pulse" / run_date
    archive_dir.mkdir(parents=True, exist_ok=True)
    pulse_archive_html = pulse_tpl.render(pulse=pulse, asset_path="../../assets", base_path="../..")
    (archive_dir / "index.html").write_text(pulse_archive_html, encoding="utf-8")
    (archive_dir / "pulse.json").write_text(pulse_json, encoding="utf-8")
    (archive_dir / "pulse.md").write_text(pulse_md, encoding="utf-8")

    pulse_index_tpl = env.get_template("pulse_index.html.j2")
    pulse_dates = sorted([p.name for p in (reports / "pulse").iterdir() if p.is_dir()])
    pulse_index_html = pulse_index_tpl.render(dates=pulse_dates, asset_path="../assets", base_path="..")
    (reports / "pulse" / "index.html").write_text(pulse_index_html, encoding="utf-8")

    print("Built report.")


if __name__ == "__main__":
    main()
