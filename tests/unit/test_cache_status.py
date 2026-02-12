import os
import pytest
from fastapi.testclient import TestClient
from src.api.server import app, _rate_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear rate-limiter state so tests aren't throttled by prior test runs."""
    _rate_limiter._requests.clear()
    yield


def test_cache_status_structure():
    os.environ["ADMIN_WARMUP_KEY"] = "test-secret"
    client = TestClient(app)
    response = client.get("/admin/cache-status", headers={"X-Admin-Key": "test-secret"})
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "database" in data
    assert "driver" in data
    assert "cache_index_entries" in data
    assert "local_cache_size_mb" in data
    assert "local_cache_files" in data
    assert "local_cache_path" in data

    # Check types
    assert isinstance(data["cache_index_entries"], int)
    assert isinstance(data["local_cache_size_mb"], (int, float))

def test_cache_status_integration():
    os.environ["ADMIN_WARMUP_KEY"] = "test-secret"
    client = TestClient(app)
    response = client.get("/admin/cache-status", headers={"X-Admin-Key": "test-secret"})
    if response.status_code == 200:
        data = response.json()
        if data["database"] == "connected":
            assert data["status"] == "ok"
        else:
            assert data["status"] == "degraded"
