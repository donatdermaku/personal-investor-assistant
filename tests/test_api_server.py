import importlib

from fastapi.testclient import TestClient


def _load_test_app(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_DB_PATH", str(tmp_path / "user.db"))
    monkeypatch.setenv("NEXUS_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("NEXUS_ALLOWED_ORIGINS", "http://localhost:3000")

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
