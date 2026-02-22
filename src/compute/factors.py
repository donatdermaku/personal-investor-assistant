from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src import industry_map
from src.utils_io import (
    PARQ,
    ROOT,
    db_conn,
    load_yaml,
    register_temp_view,
    today_str,
    unregister_temp_view,
)
from src.utils_stats import (
    calc_ev_to_ebitda,
    calc_fcf_yield,
    calc_roic,
    industry_zscores,
    pct_change_n,
    piotroski_components,
    piotroski_f_score,
    robust_zscore,
    winsorize,
    zscore,
)
from src.transform.quality import qa_checks


def _map_industry(sic: str) -> str:
    return industry_map.map_sic_to_industry(sic)


def _calc_price_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    mom12_map: dict[str, float] = {}
    if not prices.empty:
        con = duckdb.connect()
        con.register("prices_df", prices)
        mom12_df = con.execute(
            """
            with lagged as (
                select
                    ticker,
                    cast(date as date) as date,
                    cast(adj_close as double) as adj_close,
                    lag(cast(adj_close as double), 252) over (
                        partition by ticker
                        order by cast(date as date)
                    ) as lag252,
                    row_number() over (
                        partition by ticker
                        order by cast(date as date) desc
                    ) as rn
                from prices_df
                where adj_close is not null
            )
            select
                ticker,
                case
                    when lag252 is null or lag252 = 0 then null
                    else (adj_close / lag252) - 1
                end as mom12m_sql
            from lagged
            where rn = 1
            """
        ).df()
        con.unregister("prices_df")
        con.close()
        if not mom12_df.empty:
            mom12_map = {
                str(row["ticker"]): float(row["mom12m_sql"])
                for _, row in mom12_df.dropna(subset=["mom12m_sql"]).iterrows()
            }

    out_rows = []
    for ticker, grp in prices.groupby("ticker"):
        g = grp.sort_values("date")
        series = g["adj_close"].dropna()
        if series.empty:
            continue
        last = series.iloc[-1]
        mom3 = pct_change_n(series, 63)
        mom6 = pct_change_n(series, 126)
        mom12 = mom12_map.get(str(ticker), np.nan)
        ma50 = series.rolling(50).mean().iloc[-1] if len(series) >= 50 else np.nan
        ma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else np.nan
        dist_50 = last / ma50 - 1 if ma50 and ma50 == ma50 else np.nan
        dist_200 = last / ma200 - 1 if ma200 and ma200 == ma200 else np.nan
        returns = series.pct_change().dropna()
        vol_30 = returns.tail(30).std(ddof=0) * np.sqrt(252) if len(returns) >= 20 else np.nan
        vol_252 = returns.tail(252).std(ddof=0) * np.sqrt(252) if len(returns) >= 200 else np.nan
        sharpe_1y = np.nan
        if len(returns) >= 200:
            std = returns.tail(252).std(ddof=0)
            if std and std == std:
                sharpe_1y = np.sqrt(252) * returns.tail(252).mean() / std
        window = series.tail(252)
        drawdown = np.nan
        if not window.empty:
            peak = window.max()
            if peak:
                drawdown = last / peak - 1
        out_rows.append({
            "ticker": ticker,
            "Price": float(last),
            "Mom3m": mom3,
            "Mom6m": mom6,
            "Mom12m": mom12,
            "Dist50d": dist_50,
            "Dist200d": dist_200,
            "Volatility30d": vol_30,
            "Volatility1y": vol_252,
            "Sharpe1y": sharpe_1y,
            "Drawdown1y": drawdown,
            "price_count": len(series),
            "price_last_date": g["date"].iloc[-1],
        })
    return pd.DataFrame(out_rows)


def _calc_fundamental_metrics(fnds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fnds.empty:
        return pd.DataFrame(), pd.DataFrame()
    con = duckdb.connect()
    con.register("fnds_df", fnds)
    rollup = con.execute(
        """
        with ordered as (
            select
                ticker,
                cast(fiscal_end as date) as fiscal_end,
                cast(Revenue as double) as Revenue,
                cast(NetIncome as double) as NetIncome,
                cast(SharesDiluted as double) as SharesDiluted,
                cast(OperatingCF as double) as OperatingCF,
                cast(CapitalExpenditures as double) as CapitalExpenditures,
                cast(GrossProfit as double) as GrossProfit,
                cast(EBITDA as double) as EBITDA,
                cast(InterestExpense as double) as InterestExpense,
                cast(Debt as double) as Debt,
                cast(CashAndEquivalents as double) as CashAndEquivalents,
                cast(TotalAssets as double) as TotalAssets,
                cast(TotalLiabilities as double) as TotalLiabilities,
                cast(CurrentAssets as double) as CurrentAssets,
                cast(CurrentLiabilities as double) as CurrentLiabilities,
                cast(filed as timestamp) as filed,
                cast(sic as varchar) as sic,
                cast(cik as varchar) as cik,
                cast(entity_name as varchar) as entity_name
            from fnds_df
        )
        select
            ticker,
            fiscal_end,
            sum(Revenue) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as RevenueTTM,
            sum(NetIncome) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as NetIncomeTTM,
            avg(SharesDiluted) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as SharesDilutedTTM,
            sum(OperatingCF) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as OpCFTTM,
            sum(CapitalExpenditures) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as CapexTTM,
            sum(GrossProfit) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as GrossProfitTTM,
            sum(EBITDA) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as EBITDATTM,
            sum(InterestExpense) over (
                partition by ticker
                order by fiscal_end
                rows between 3 preceding and current row
            ) as InterestExpenseTTM,
            Debt,
            CashAndEquivalents,
            TotalAssets,
            TotalLiabilities,
            CurrentAssets,
            CurrentLiabilities,
            filed,
            sic,
            cik,
            entity_name
        from ordered
        order by ticker, fiscal_end
        """
    ).df()
    con.unregister("fnds_df")
    con.close()
    rollup["FCFTTM"] = rollup["OpCFTTM"] - rollup["CapexTTM"].fillna(0)

    piotroski_series = []
    piotroski_components_series = []
    for ticker, grp in rollup.groupby("ticker"):
        scores = piotroski_f_score(grp)
        comps = piotroski_components(grp)
        piotroski_series.append(pd.Series(scores.values, index=grp.index))
        piotroski_components_series.append(comps)

    if piotroski_series:
        rollup["PiotroskiF"] = pd.concat(piotroski_series).sort_index()
    else:
        rollup["PiotroskiF"] = np.nan
    if piotroski_components_series:
        all_comps = pd.concat(piotroski_components_series).sort_index()
        for col in all_comps.columns:
            rollup[f"Pio_{col}"] = all_comps[col].values

    rollup["EPS_TTM"] = rollup["NetIncomeTTM"] / rollup["SharesDilutedTTM"].replace({0: np.nan})
    rollup["RevenueGrowthYoY"] = rollup.groupby("ticker")["RevenueTTM"].pct_change(4)
    rollup["EPSGrowthYoY"] = rollup.groupby("ticker")["EPS_TTM"].pct_change(4)

    rollup["GrossMargin"] = rollup["GrossProfitTTM"] / rollup["RevenueTTM"].replace({0: np.nan})
    rollup["OpMargin"] = rollup["EBITDATTM"] / rollup["RevenueTTM"].replace({0: np.nan})
    rollup["FCFMargin"] = rollup["FCFTTM"] / rollup["RevenueTTM"].replace({0: np.nan})
    rollup["NetMargin"] = rollup["NetIncomeTTM"] / rollup["RevenueTTM"].replace({0: np.nan})

    rollup["GrossMarginTrend"] = rollup.groupby("ticker")["GrossMargin"].diff(4)
    rollup["OpMarginTrend"] = rollup.groupby("ticker")["OpMargin"].diff(4)
    rollup["FCFMarginTrend"] = rollup.groupby("ticker")["FCFMargin"].diff(4)

    rollup["DebtToEBITDA"] = rollup["Debt"] / rollup["EBITDATTM"].replace({0: np.nan})
    rollup["NetDebtToEBITDA"] = (rollup["Debt"].fillna(0) - rollup["CashAndEquivalents"].fillna(0)) / rollup[
        "EBITDATTM"
    ].replace({0: np.nan})
    rollup["CurrentRatio"] = rollup["CurrentAssets"] / rollup["CurrentLiabilities"].replace({0: np.nan})
    rollup["InterestCoverage"] = rollup["EBITDATTM"] / rollup["InterestExpenseTTM"].replace({0: np.nan})

    rollup["RevenueGrowthVol"] = (
        rollup.groupby("ticker")["RevenueGrowthYoY"]
        .rolling(8, min_periods=4)
        .std()
        .reset_index(level=0, drop=True)
    )
    rollup["EarningsGrowthVol"] = (
        rollup.groupby("ticker")["EPSGrowthYoY"]
        .rolling(8, min_periods=4)
        .std()
        .reset_index(level=0, drop=True)
    )

    latest = rollup.groupby("ticker").tail(1).copy()
    return rollup, latest


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


def _zs(series: pd.Series, robust: bool) -> pd.Series:
    cleaned = winsorize(series)
    return robust_zscore(cleaned) if robust else zscore(cleaned)


def compute(prices: pd.DataFrame, fnds: pd.DataFrame, universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper()

    prices = prices.copy()
    if not prices.empty:
        prices = prices[prices["ticker"].isin(universe["ticker"])].copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values(["ticker", "date"])

    fnds = fnds.copy()
    if not fnds.empty:
        fnds = fnds[fnds["ticker"].isin(universe["ticker"])].copy()

    price_metrics = _calc_price_metrics(prices) if not prices.empty else pd.DataFrame()
    rollup, fundamentals_latest = _calc_fundamental_metrics(fnds)

    base = universe.merge(fundamentals_latest, on="ticker", how="left")
    if not price_metrics.empty:
        base = base.merge(price_metrics, on="ticker", how="left")

    base["EPS_TTM"] = base["NetIncomeTTM"] / base["SharesDilutedTTM"].replace({0: np.nan})
    base["PE_TTM"] = base["Price"] / base["EPS_TTM"].replace({0: np.nan})
    base["FCFYield_TTM"] = calc_fcf_yield(base["FCFTTM"], base["Price"], base["SharesDilutedTTM"])
    base["EVToEBITDA"] = calc_ev_to_ebitda(
        base["Price"],
        base["SharesDilutedTTM"],
        base["Debt"],
        base["CashAndEquivalents"],
        base["EBITDATTM"],
    )
    base["ROIC"] = calc_roic(
        base["NetIncomeTTM"],
        base["InterestExpenseTTM"],
        base["TotalAssets"],
        base["CurrentLiabilities"],
        base["CashAndEquivalents"],
        base["Debt"],
    )
    base["Leverage"] = (
        (base["Debt"].fillna(0) - base["CashAndEquivalents"].fillna(0))
        / base["TotalAssets"].replace({0: np.nan})
    )
    base["Qual_ROA"] = base["NetIncomeTTM"] / base["TotalAssets"].replace({0: np.nan})

    base["industry"] = base["sic"].apply(_map_industry)

    uni_cfg = cfg.get("universe", {}) if cfg else {}
    min_size = int(uni_cfg.get("min_size", 200))
    use_robust = len(base) < min_size

    pe_positive = base["PE_TTM"].where(base["PE_TTM"] > 0)
    base["Value_PE"] = -_zs(np.log(pe_positive), use_robust)
    base["Value_FCF"] = _zs(base["FCFYield_TTM"], use_robust)
    evebitda_positive = base["EVToEBITDA"].where(base["EVToEBITDA"] > 0)
    base["Value_EVEBITDA"] = -_zs(np.log(evebitda_positive), use_robust)
    base["ValueScore_raw"] = base[["Value_PE", "Value_FCF", "Value_EVEBITDA"]].mean(axis=1, skipna=True)

    base["Quality_ROIC"] = _zs(base["ROIC"], use_robust)
    base["Quality_Pio"] = _zs(base["PiotroskiF"], use_robust)
    base["Quality_Margin"] = _zs(base["GrossMargin"], use_robust)
    base["Quality_Stability"] = -_zs(base["RevenueGrowthVol"], use_robust)
    base["QualityScore_raw"] = base[["Quality_ROIC", "Quality_Pio", "Quality_Margin", "Quality_Stability"]].mean(
        axis=1, skipna=True
    )

    base["Mom3m_z"] = _zs(base["Mom3m"], use_robust)
    base["Mom6m_z"] = _zs(base["Mom6m"], use_robust)
    base["Mom12m_z"] = _zs(base["Mom12m"], use_robust)
    base["MomTrend_z"] = _zs(base["Dist50d"], use_robust)
    base["MomScore_raw"] = base[["Mom3m_z", "Mom6m_z", "Mom12m_z", "MomTrend_z"]].mean(axis=1, skipna=True)

    base["ValueScore"] = industry_zscores(base["ValueScore_raw"], base["industry"], robust=use_robust)
    base["QualityScore"] = industry_zscores(base["QualityScore_raw"], base["industry"], robust=use_robust)
    base["MomScore"] = _zs(base["MomScore_raw"], use_robust)

    weights_cfg = cfg.get("weights", {}) if cfg else {}
    default_weights = {"value": 0.4, "quality": 0.4, "momentum": 0.2}
    weights = {
        "value": float(weights_cfg.get("value", default_weights["value"])),
        "quality": float(weights_cfg.get("quality", default_weights["quality"])),
        "momentum": float(weights_cfg.get("momentum", default_weights["momentum"])),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = default_weights
        total_weight = 1.0
    weights = {k: v / total_weight for k, v in weights.items()}

    base["Composite"] = (
        weights["value"] * base["ValueScore"]
        + weights["quality"] * base["QualityScore"]
        + weights["momentum"] * base["MomScore"]
    )

    base["value_pct"] = _pct_rank(base["ValueScore"])
    base["quality_pct"] = _pct_rank(base["QualityScore"])
    base["momentum_pct"] = _pct_rank(base["MomScore"])
    base["composite_pct"] = _pct_rank(base["Composite"])

    base["composite_contrib_value"] = weights["value"] * base["ValueScore"]
    base["composite_contrib_quality"] = weights["quality"] * base["QualityScore"]
    base["composite_contrib_momentum"] = weights["momentum"] * base["MomScore"]

    base["has_prices"] = base["price_count"].fillna(0) >= 200
    base["has_fundamentals"] = base["RevenueTTM"].notna() | base["NetIncomeTTM"].notna()

    staleness = []
    for _, row in base.iterrows():
        filed = row.get("filed")
        fiscal_end = row.get("fiscal_end")
        latest = filed if pd.notna(filed) else fiscal_end
        if pd.isna(latest):
            staleness.append(np.nan)
        else:
            dt = pd.to_datetime(latest)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            staleness.append((pd.Timestamp.now(tz=timezone.utc) - dt).days)
    base["fundamentals_staleness_days"] = staleness

    key_fields = ["Price", "RevenueTTM", "NetIncomeTTM", "SharesDilutedTTM", "EBITDATTM"]
    missing = []
    for _, row in base.iterrows():
        missing_fields = [k for k in key_fields if pd.isna(row.get(k))]
        missing.append(",".join(missing_fields))
    base["missing_key_fields"] = missing

    base["ValueZ"] = base["ValueScore"]
    base["QualityZ"] = base["QualityScore"]

    cols = [
        "ticker",
        "Price",
        "PE_TTM",
        "FCFYield_TTM",
        "EVToEBITDA",
        "ROIC",
        "Leverage",
        "Qual_ROA",
        "PiotroskiF",
        "Mom3m",
        "Mom6m",
        "Mom12m",
        "Dist50d",
        "Dist200d",
        "Drawdown1y",
        "Volatility30d",
        "Volatility1y",
        "Sharpe1y",
        "ValueScore",
        "QualityScore",
        "MomScore",
        "Composite",
        "value_pct",
        "quality_pct",
        "momentum_pct",
        "composite_pct",
        "composite_contrib_value",
        "composite_contrib_quality",
        "composite_contrib_momentum",
        "ValueZ",
        "QualityZ",
        "GrossMargin",
        "OpMargin",
        "FCFMargin",
        "RevenueGrowthYoY",
        "EPSGrowthYoY",
        "GrossMarginTrend",
        "OpMarginTrend",
        "FCFMarginTrend",
        "DebtToEBITDA",
        "NetDebtToEBITDA",
        "CurrentRatio",
        "InterestCoverage",
        "RevenueGrowthVol",
        "EarningsGrowthVol",
        "industry",
        "sic",
        "filed",
        "fiscal_end",
        "cik",
        "entity_name",
        "has_prices",
        "has_fundamentals",
        "fundamentals_staleness_days",
        "missing_key_fields",
    ]

    for col in [c for c in base.columns if c.startswith("Pio_")]:
        cols.append(col)

    return base[cols].sort_values("Composite", ascending=False)


def _find_prior_scores(history_paths: list[pathlib.Path], target_date: datetime) -> pd.DataFrame:
    candidates = []
    for path in history_paths:
        try:
            date_part = path.stem.split("_")[-1]
            snap_date = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        if snap_date <= target_date:
            candidates.append((snap_date, path))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda x: x[0])
    return pd.read_parquet(candidates[-1][1])


def main() -> None:
    con = db_conn()
    uni_path = ROOT / "data" / "universe.csv"
    universe = pd.read_csv(uni_path) if uni_path.exists() else pd.DataFrame(columns=["ticker"])
    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    tickers = universe["ticker"].dropna().unique().tolist()

    prices_cols = ["ticker", "date", "adj_close"]
    fnds_cols = [
        "ticker",
        "fiscal_end",
        "Revenue",
        "NetIncome",
        "SharesDiluted",
        "OperatingCF",
        "CapitalExpenditures",
        "GrossProfit",
        "EBITDA",
        "InterestExpense",
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
    ]

    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        prices = con.execute(
            f"""
            SELECT ticker, date, adj_close
            FROM prices_daily
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, date
            """,
            tickers,
        ).df()
        fnds = con.execute(
            f"""
            SELECT {",".join(fnds_cols)}
            FROM fundamentals_quarterly
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, fiscal_end
            """,
            tickers,
        ).df()
    else:
        prices = pd.DataFrame(columns=prices_cols)
        fnds = pd.DataFrame(columns=fnds_cols)

    cfg = load_yaml(ROOT / "config.yml") or {}
    scores = compute(prices, fnds, universe, cfg)

    today = today_str()
    history_paths = sorted(PARQ.glob("scores_daily_*.parquet"))
    if not scores.empty:
        scores["composite_pct_change_1d"] = np.nan
        scores["composite_pct_change_7d"] = np.nan
        if history_paths:
            today_dt = datetime.strptime(today, "%Y-%m-%d")
            prior_1d = _find_prior_scores(history_paths, today_dt)
            prior_7d = _find_prior_scores(history_paths, today_dt - pd.Timedelta(days=7))
            if not prior_1d.empty:
                prior_1d = prior_1d.set_index("ticker")
                scores = scores.set_index("ticker")
                scores["composite_pct_change_1d"] = scores["composite_pct"] - prior_1d["composite_pct"]
                scores = scores.reset_index()
            if not prior_7d.empty:
                prior_7d = prior_7d.set_index("ticker")
                scores = scores.set_index("ticker")
                scores["composite_pct_change_7d"] = scores["composite_pct"] - prior_7d["composite_pct"]
                scores = scores.reset_index()

    scores.to_parquet(PARQ / f"scores_daily_{today}.parquet", index=False)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scores_daily (
            ticker VARCHAR,
            Price DOUBLE,
            PE_TTM DOUBLE,
            FCFYield_TTM DOUBLE,
            EVToEBITDA DOUBLE,
            ROIC DOUBLE,
            Leverage DOUBLE,
            Qual_ROA DOUBLE,
            PiotroskiF DOUBLE,
            Mom3m DOUBLE,
            Mom6m DOUBLE,
            Mom12m DOUBLE,
            Dist50d DOUBLE,
            Dist200d DOUBLE,
            Drawdown1y DOUBLE,
            Volatility30d DOUBLE,
            Volatility1y DOUBLE,
            Sharpe1y DOUBLE,
            ValueScore DOUBLE,
            QualityScore DOUBLE,
            MomScore DOUBLE,
            Composite DOUBLE,
            value_pct DOUBLE,
            quality_pct DOUBLE,
            momentum_pct DOUBLE,
            composite_pct DOUBLE,
            composite_contrib_value DOUBLE,
            composite_contrib_quality DOUBLE,
            composite_contrib_momentum DOUBLE,
            ValueZ DOUBLE,
            QualityZ DOUBLE,
            GrossMargin DOUBLE,
            OpMargin DOUBLE,
            FCFMargin DOUBLE,
            RevenueGrowthYoY DOUBLE,
            EPSGrowthYoY DOUBLE,
            GrossMarginTrend DOUBLE,
            OpMarginTrend DOUBLE,
            FCFMarginTrend DOUBLE,
            DebtToEBITDA DOUBLE,
            NetDebtToEBITDA DOUBLE,
            CurrentRatio DOUBLE,
            InterestCoverage DOUBLE,
            RevenueGrowthVol DOUBLE,
            EarningsGrowthVol DOUBLE,
            industry VARCHAR,
            sic VARCHAR,
            filed TIMESTAMP,
            fiscal_end DATE,
            cik VARCHAR,
            entity_name VARCHAR,
            has_prices BOOLEAN,
            has_fundamentals BOOLEAN,
            fundamentals_staleness_days DOUBLE,
            missing_key_fields VARCHAR,
            composite_pct_change_1d DOUBLE,
            composite_pct_change_7d DOUBLE
        )
        """
    )
    con.execute("DELETE FROM scores_daily")
    view_name = register_temp_view(con, "scores_tmp", scores)
    if view_name:
        con.execute("DROP TABLE IF EXISTS scores_daily")
        con.execute(
            f"CREATE TABLE scores_daily AS SELECT * FROM {view_name} WHERE 1=0"
        )
        con.execute(
            f"INSERT INTO scores_daily SELECT * FROM {view_name}"
        )
    unregister_temp_view(con, view_name)


    uni_cfg = cfg.get("universe", {}) if cfg else {}
    min_size = int(uni_cfg.get("min_size", 200))
    qa = qa_checks(scores, min_size)
    qa_path = ROOT / "data" / "qa"
    qa_path.mkdir(parents=True, exist_ok=True)
    (qa_path / f"qa_{today}.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    for warning in qa["warnings"]:
        print(f"[WARN] {warning}")
    print("Computed scores:", len(scores))


if __name__ == "__main__":
    main()
