from __future__ import annotations

from pathlib import Path
from typing import Iterable


def export_summary_html(path: Path, sections: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join([f"<h2>{title}</h2><p>{content}</p>" for title, content in sections])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Summary</title></head><body>{body}</body></html>"
    path.write_text(html, encoding="utf-8")
