import importlib
import os
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient
from fastapi import HTTPException


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
    from src.api.server import parse_allowed_origins, resolve_allowed_origins

    assert parse_allowed_origins(None) == ["http://localhost:3000"]
    assert parse_allowed_origins("") == ["http://localhost:3000"]
    assert parse_allowed_origins("https://example.com") == ["https://example.com"]
    assert parse_allowed_origins("http://example.com") == []
    assert parse_allowed_origins("http://localhost:3000, https://app.example.com") == [
        "http://localhost:3000",
        "https://app.example.com",
    ]
    assert resolve_allowed_origins(None, "production") == []
    assert resolve_allowed_origins("https://app.example.com", "production") == ["https://app.example.com"]


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


def test_ops_health_uses_latest_run_id_fallback(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(
        server.repo,
        "get_latest_run",
        lambda: SimpleNamespace(
            id="run-from-id-field",
            status="completed",
            created_at=datetime.now(timezone.utc),
        ),
    )

    response = client.get("/ops/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_run"]["run_id"] == "run-from-id-field"


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


def test_rate_limit_is_user_scoped_in_supabase_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("NEXUS_RATE_LIMIT_PER_WINDOW", "1")
    monkeypatch.setenv("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60")
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.setattr(server.repo, "list_runs", lambda limit=50, user_id=None: [])

    token_a = _make_hs256_jwt({"sub": "user-a", "exp": int(time.time()) + 600}, "test-secret")
    token_b = _make_hs256_jwt({"sub": "user-b", "exp": int(time.time()) + 600}, "test-secret")

    first_a = client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token_a}"})
    second_a = client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token_a}"})
    first_b = client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token_b}"})

    assert first_a.status_code == 200
    assert second_a.status_code == 429
    assert first_b.status_code == 200


def test_report_pdf_endpoint(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "_load_report_html", lambda run_id, user_id=None: "<html><body>ok</body></html>")
    monkeypatch.setattr(server, "html_to_pdf_bytes", lambda html, base_url=None: b"%PDF-1.7\nfake")

    response = client.get("/api/reports/test-run/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")


def test_report_pdf_endpoint_not_found(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    def _raise_not_found(run_id: str, user_id: str | None = None):
        raise HTTPException(status_code=404, detail="Report not found")

    monkeypatch.setattr(server, "_load_report_html", _raise_not_found)
    response = client.get("/api/reports/test-run/pdf")
    assert response.status_code == 404


def _make_hs256_jwt(payload: dict, secret: str, alg: str = "HS256") -> str:
    header = {"alg": alg, "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("utf-8").rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def _make_dummy_jwt(payload: dict, alg: str = "RS256") -> str:
    header = {"alg": alg, "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("utf-8").rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    signature_b64 = base64.urlsafe_b64encode(b"sig").decode("utf-8").rstrip("=")
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def test_decode_and_verify_hs256_jwt_valid_token() -> None:
    from src.api.server import _decode_and_verify_hs256_jwt

    secret = "test-secret"
    payload = {"sub": "user-123", "exp": int(time.time()) + 3600}
    token = _make_hs256_jwt(payload, secret)
    decoded = _decode_and_verify_hs256_jwt(token, secret)
    assert decoded["sub"] == "user-123"


def test_decode_and_verify_hs256_jwt_invalid_signature() -> None:
    from src.api.server import _decode_and_verify_hs256_jwt

    secret = "test-secret"
    payload = {"sub": "user-123", "exp": int(time.time()) + 3600}
    token = _make_hs256_jwt(payload, "other-secret")
    try:
        _decode_and_verify_hs256_jwt(token, secret)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected HTTPException for invalid JWT signature.")


def test_decode_and_verify_hs256_jwt_rejects_non_hs256_header() -> None:
    from src.api.server import _decode_and_verify_hs256_jwt

    secret = "test-secret"
    payload = {"sub": "user-123", "exp": int(time.time()) + 3600}
    token = _make_hs256_jwt(payload, secret, alg="RS256")
    try:
        _decode_and_verify_hs256_jwt(token, secret)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "algorithm" in exc.detail.lower()
    else:
        raise AssertionError("Expected HTTPException for unsupported JWT algorithm.")


def test_supabase_endpoints_require_bearer_token(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")

    response = client.get("/api/v1/runs")
    assert response.status_code == 401
    assert "bearer" in response.json().get("detail", "").lower()


def test_list_runs_scoped_by_authenticated_user(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    secret = "test-secret"
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    token = _make_hs256_jwt({"sub": "user-42", "exp": int(time.time()) + 600}, secret)

    captured_user_id = {"value": None}

    def _fake_list_runs(limit: int = 50, user_id: str | None = None):
        captured_user_id["value"] = user_id
        return [
            SimpleNamespace(
                id="run-abc",
                status="completed",
                completed_at=datetime.now(timezone.utc),
            )
        ]

    monkeypatch.setattr(server.repo, "list_runs", _fake_list_runs)
    response = client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert captured_user_id["value"] == "user-42"
    payload = response.json()
    assert payload["runs"][0]["run_id"] == "run-abc"


def test_list_runs_scoped_by_authenticated_user_with_non_hs256_token(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"id": "user-rs256", "email": "user@example.com"}

    monkeypatch.setattr(server.requests, "get", lambda *args, **kwargs: _Resp())
    token = _make_dummy_jwt({"sub": "ignored-by-remote-check"}, alg="RS256")

    captured_user_id = {"value": None}

    def _fake_list_runs(limit: int = 50, user_id: str | None = None):
        captured_user_id["value"] = user_id
        return []

    monkeypatch.setattr(server.repo, "list_runs", _fake_list_runs)
    response = client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert captured_user_id["value"] == "user-rs256"


def test_run_summary_denies_access_when_not_owned(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    secret = "test-secret"
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    token = _make_hs256_jwt({"sub": "user-1", "exp": int(time.time()) + 600}, secret)

    monkeypatch.setattr(server.repo, "get_run_by_id", lambda run_id, user_id=None: None)
    response = client.get("/api/v1/run/run-123/summary", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404
    assert response.json().get("detail") == "Run not found"


def test_export_artifact_passes_user_scope_to_repo(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    secret = "test-secret"
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    token = _make_hs256_jwt({"sub": "user-9", "exp": int(time.time()) + 600}, secret)

    monkeypatch.setattr(server.repo, "get_run_by_id", lambda run_id, user_id=None: SimpleNamespace(id=run_id))
    captured = {"user_id": None, "filename": None}

    def _fake_get_artifact_bytes(run_id: str, filename: str, user_id: str | None = None):
        captured["user_id"] = user_id
        captured["filename"] = filename
        return b"{}", "application/json"

    monkeypatch.setattr(server.repo, "get_artifact_bytes", _fake_get_artifact_bytes)

    response = client.get(
        "/api/v1/run/run-xyz/export/summary-json",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert captured["user_id"] == "user-9"
    assert captured["filename"] == "summary.json"


def test_create_portfolio_endpoint_local_mode(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post("/api/v1/portfolios", json={"name": "Retirement", "currency": "USD"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["portfolio"]["id"] > 0
    assert payload["portfolio"]["name"] == "Retirement"
    assert payload["portfolio"]["currency"] == "USD"


def test_create_portfolio_requires_auth_in_supabase_mode(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setattr(server, "use_supabase", lambda: True)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret")

    response = client.post("/api/v1/portfolios", json={"name": "Growth"})
    assert response.status_code == 401


def test_admin_error_events_requires_admin_key(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/admin/error-events")
    assert response.status_code == 403


def test_admin_error_events_returns_recent_entries(tmp_path, monkeypatch) -> None:
    app = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app)
    import src.api.server as server

    monkeypatch.setenv("ADMIN_WARMUP_KEY", "secret")
    server._record_error_event(
        request=SimpleNamespace(url=SimpleNamespace(path="/api/v1/run/bad"), method="GET"),
        status=500,
        error_code="TEST_FAILURE",
        message="simulated failure",
    )
    response = client.get("/admin/error-events?limit=5", headers={"x-admin-key": "secret"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert payload["events"][0]["error_code"] in {"TEST_FAILURE", "HTTP_5XX", "UNHANDLED_EXCEPTION"}
