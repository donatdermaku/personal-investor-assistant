from __future__ import annotations

import pandas as pd

from market_data.persistent_cache import load_cached_frame, store_cached_frame
from storage import db, models


def test_persistent_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_MARKET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NEXUS_DB_PATH", str(tmp_path / "cache.db"))
    engine = db.init_db(str(tmp_path / "cache.db"))
    models.Base.metadata.create_all(bind=engine)

    frame = pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]})
    store_cached_frame(
        source="fred",
        key="TEST",
        frame=frame,
        ttl_seconds=3600,
        asof_date="2024-01-01",
        coverage_pct=1.0,
        status="fresh",
    )
    result = load_cached_frame("fred", "TEST")
    assert not result.frame.empty
    assert result.status in ("fresh", "stale")
