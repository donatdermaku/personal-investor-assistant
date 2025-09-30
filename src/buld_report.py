import pathlib, pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.utils_io import ROOT, PARQ, today_str

def main():
    # load latest parquet snapshots by prefix
    def latest(prefix: str) -> pathlib.Path:
        files = sorted((PARQ).glob(f"{prefix}_*.parquet"))
        return files[-1] if files else None

    prices_p = latest("prices_daily")
    fnds_p = latest("fundamentals_quarterly")
    scores_p = latest("scores_daily")

    scores = pd.read_parquet(scores_p) if scores_p else pd.DataFrame()
    top10 = scores.head(10).round(3)

    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        autoescape=select_autoescape()
    )
    tpl = env.get_template("report.html.j2")
    html = tpl.render(
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        top10=top10.to_dict(orient="records")
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "latest.html").write_text(html, encoding="utf-8")

    # monthly snapshot
    ym = datetime.utcnow().strftime("%Y-%m")
    (reports / f"{ym}").mkdir(parents=True, exist_ok=True)
    (reports / f"{ym}/index.html").write_text(html, encoding="utf-8")
    print("Built report.")

if __name__ == "__main__":
    main()
