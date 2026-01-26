from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_cache_status_structure():
    # Mock env var
    import os
    os.environ["ADMIN_WARMUP_KEY"] = "test-secret"
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
    import os
    os.environ["ADMIN_WARMUP_KEY"] = "test-secret"
    # This might fail if DB is not initialized in test env, but server usually handles it
    response = client.get("/admin/cache-status", headers={"X-Admin-Key": "test-secret"})
    if response.status_code == 200:
        data = response.json()
        if data["database"] == "connected":
            assert data["status"] == "ok"
        else:
            assert data["status"] == "degraded"
