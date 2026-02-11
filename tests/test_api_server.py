import importlib
import os

from fastapi.testclient import TestClient


def _load_test_app(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_DB_PATH", str(tmp_path / "user.db"))
    monkeypatch.setenv("NEXUS_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("NEXUS_ALLOWED_ORIGINS", "http://localhost:3000")
    if os.getenv("NEXUS_RATE_LIMIT_ENABLED") is None:
        monkeypatch.setenv("NEXUS_RATE_LIMIT_ENABLED", "1")
    if os.getenv("NEXUS_RATE_LIMIT_PER_WINDOW") is None:
        monkeypatch.setenv("NEXUS_RATE_LIMIT_PER_WINDOW", "120")
    if os.getenv("NEXUS_RATE_LIMIT_WINDOW_SECONDS") is None:
        monkeypatch.setenv("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60")

    import storage.db as db
    import storage.repo as repo
    import storage.datamanager as datamanager
    import src.api.server as server
    import storage.models as models

    importlib.reload(db)
    importlib.reload(repo)
    importlib.reload(datamanager)
    importlib.reload(server)

    models.Base.metadata.create_all(bind=db.get_engine())
    return server.app


def test_parse_allowed_origins() -> None:
    from src.api.server import parse_allowed_origins

    assert parse_allowed_origins(None) == ["http://localhost:3000"]
    assert parse_allowed_origins("") == ["http://localhost:3000"]
    assert parse_allowed_origins("https://example.com") == ["https://example.com"]
    assert parse_allowed_origins("http://example.com") == []
    assert parse_allowed_origins("http://localhost:3000, https://app.example.com") == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_health_endpoint(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_latest_run_empty(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.get("/latest-run")
    assert response.status_code == 404
    payload = response.json()
    assert "No runs found" in payload.get("detail", "")


def test_ops_health_endpoint(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.get("/ops/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "runtime" in payload
    assert "database" in payload
    assert "rate_limit" in payload


def test_rate_limit_enforced(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("NEXUS_RATE_LIMIT_PER_WINDOW", "2")
    monkeypatch.setenv("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60")
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    first = client.get("/latest-run")
    second = client.get("/latest-run")
    third = client.get("/latest-run")

    assert first.status_code == 404
    assert second.status_code == 404
    assert third.status_code == 429
    assert "retry" in str(third.json()).lower()
