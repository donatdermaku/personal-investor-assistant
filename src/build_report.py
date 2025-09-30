import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils_io import ROOT, PARQ, load_yaml


def _latest(prefix: str) -> pathlib.Path | None:
    files = sorted(PARQ.glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def _all(prefix: str) -> list[pathlib.Path]:
    return sorted(PARQ.glob(f"{prefix}_*.parquet"))


def _safe_read_parquet(path: pathlib.Path) -> pd.DataFrame:
    """Read a parquet file and return an empty DataFrame on failure, logging a warning."""
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover - environment dependent
        # Import here to avoid top-level dependency for lightweight runs
        import sys

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


def main():
    watch = load_yaml(ROOT / "watchlist.yml")
    tickers = watch.get("tickers", [])
    notes = watch.get("notes", {})
    weights = _normalize_weights(tickers, watch.get("weights", {}))

    scores_path = _latest("scores_daily")
    fundamentals_path = _latest("fundamentals_quarterly")
    prices_path = _latest("prices_daily")

    scores = _safe_read_parquet(scores_path) if scores_path else pd.DataFrame()
    fundamentals = _safe_read_parquet(fundamentals_path) if fundamentals_path else pd.DataFrame()
    prices = _safe_read_parquet(prices_path) if prices_path else pd.DataFrame()

    scores = scores[scores["ticker"].isin(tickers)] if not scores.empty else scores
    fundamentals = fundamentals[fundamentals["ticker"].isin(tickers)] if not fundamentals.empty else fundamentals
    prices = prices[prices["ticker"].isin(tickers)] if not prices.empty else prices

    last_fundamental = pd.DataFrame()
    if not fundamentals.empty:
        fundamentals["fiscal_end"] = pd.to_datetime(fundamentals["fiscal_end"])
        if "filed" in fundamentals.columns:
            fundamentals["filed"] = pd.to_datetime(fundamentals["filed"])
        last_fundamental = fundamentals.sort_values("fiscal_end").groupby("ticker").tail(1)

    last_scores = scores.set_index("ticker") if not scores.empty else pd.DataFrame()
    prior_scores = pd.DataFrame()
    history_paths = _all("scores_daily")
    if len(history_paths) >= 2:
        prior_scores = pd.read_parquet(history_paths[-2])
        prior_scores = prior_scores[prior_scores["ticker"].isin(tickers)].set_index("ticker")

    composite_history = []
    for path in history_paths:
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

    latest_portfolio_composite = np.nan
    if not scores.empty and not weights.empty:
        latest_comp = scores.set_index("ticker")["Composite"].reindex(weights.index).fillna(0.0)
        latest_portfolio_composite = float((latest_comp * weights).sum())

    price_history = []
    portfolio_drawdown = np.nan
    portfolio_return = np.nan
    if not prices.empty:
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

    watch_rows = []
    for t in tickers:
        row = last_scores.loc[t] if t in last_scores.index else None
        prev = prior_scores.loc[t] if not prior_scores.empty and t in prior_scores.index else None
        fund = (
            last_fundamental[last_fundamental["ticker"] == t].iloc[0]
            if not last_fundamental.empty and t in last_fundamental["ticker"].values
            else None
        )
        filing_url = None
        filed = None
        if fund is not None:
            filed = fund.get("filed") or fund.get("fiscal_end")
            cik = str(fund.get("cik") or "").zfill(10) if fund.get("cik") else ""
            if cik:
                filing_url = f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude"
        if row is not None and filed is None:
            filed = row.get("filed")
        composite_delta = None
        if row is not None and prev is not None:
            composite_delta = float(row.get("Composite", np.nan) - prev.get("Composite", np.nan))
        watch_rows.append({
            "ticker": t,
            "note": notes.get(t, ""),
            "price": None if row is None else float(row.get("Price", np.nan)),
            "composite": None if row is None else float(row.get("Composite", np.nan)),
            "value_z": None if row is None else float(row.get("ValueZ", np.nan)),
            "quality_z": None if row is None else float(row.get("QualityZ", np.nan)),
            "momentum": None if row is None else float(row.get("MomScore", np.nan)),
            "piotroski": None if row is None else float(row.get("PiotroskiF", np.nan)),
            "volatility": None if row is None else float(row.get("Volatility30d", np.nan)),
            "sharpe": None if row is None else float(row.get("Sharpe1y", np.nan)),
            "industry": None if row is None else row.get("industry"),
            "filed": filed.strftime("%Y-%m-%d") if isinstance(filed, pd.Timestamp) else filed,
            "filing_url": filing_url,
            "composite_delta": composite_delta,
            "weight": float(weights.get(t, 0.0)),
        })

    warnings = []
    if not prices.empty:
        latest_price_date = pd.to_datetime(prices["date"]).max()
        if latest_price_date is not None:
            # compute lag in a timezone-safe way: convert both to UTC-aware and normalize
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            try:
                if latest_price_date.tzinfo is None and getattr(latest_price_date, "tz", None) is None:
                    latest_norm = latest_price_date.tz_localize("UTC").normalize()
                else:
                    latest_norm = latest_price_date.tz_convert("UTC").normalize()
            except Exception:
                # fallback to naive normalize if conversion fails
                latest_norm = pd.to_datetime(latest_price_date).normalize()

            lag_days = (now_utc - latest_norm).days
            if lag_days > 3:
                warnings.append(f"Prices are {lag_days} days old.")
    if not fundamentals.empty:
        latest_fund_date = pd.to_datetime(fundamentals["fiscal_end"]).max()
        if latest_fund_date is not None:
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            try:
                if latest_fund_date.tzinfo is None and getattr(latest_fund_date, "tz", None) is None:
                    fund_norm = latest_fund_date.tz_localize("UTC").normalize()
                else:
                    fund_norm = latest_fund_date.tz_convert("UTC").normalize()
            except Exception:
                fund_norm = pd.to_datetime(latest_fund_date).normalize()

            lag_quarters = (now_utc - fund_norm).days / 90
            if lag_quarters > 2:
                warnings.append("Fundamentals stale (>2 quarters).")
    if not watch_rows:
        warnings.append("No tickers available in latest scores snapshot.")

    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        autoescape=select_autoescape(),
    )
    tpl = env.get_template("report.html.j2")
    html = tpl.render(
    generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        watch_rows=watch_rows,
        composite_history=composite_history,
        price_history=price_history,
        portfolio_return=portfolio_return,
        portfolio_drawdown=portfolio_drawdown,
        warnings=warnings,
        latest_portfolio_composite=latest_portfolio_composite,
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "latest.html").write_text(html, encoding="utf-8")
    (reports / "index.html").write_text(html, encoding="utf-8")

    ym = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    (reports / f"{ym}").mkdir(parents=True, exist_ok=True)
    (reports / f"{ym}/index.html").write_text(html, encoding="utf-8")
    print("Built report.")

if __name__ == "__main__":
    main()
