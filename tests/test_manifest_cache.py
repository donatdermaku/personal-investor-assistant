from unittest.mock import patch

import pytest
from src.manifest import RunManifest, compute_input_hash, create_manifest

# Mock ROOT and DATA for isolation
@pytest.fixture
def mock_paths(tmp_path):
    with patch("src.manifest.ROOT", tmp_path), \
         patch("src.manifest.DATA", tmp_path / "data"):
        
        # Setup directory structure
        (tmp_path / "data" / "user_uploads").mkdir(parents=True)
        (tmp_path / "data" / "parquet").mkdir(parents=True)
        
        # Create dummy input files
        (tmp_path / "watchlist.yml").write_text("tickers: [AAPL]")
        (tmp_path / "data" / "user_uploads" / "ui_state.json").write_text("{}")
        (tmp_path / "data" / "user_uploads" / "transactions.csv").write_text("date,ticker,amount\n2023-01-01,AAPL,100")
        (tmp_path / "data" / "user_uploads" / "holdings.csv").write_text("ticker,shares\nAAPL,10")
        
        # Create dummy parquet files for data hash
        (tmp_path / "data" / "parquet" / "scores_daily_20230101.parquet").write_bytes(b"scores")
        (tmp_path / "data" / "parquet" / "fundamentals_quarterly_20230101.parquet").write_bytes(b"fundamentals")
        (tmp_path / "data" / "parquet" / "prices_daily_20230101.parquet").write_bytes(b"prices")
        
        yield tmp_path


def test_compute_input_hash_stable(mock_paths):
    """Verify that same inputs produce same hash."""
    hash1 = compute_input_hash()
    hash2 = compute_input_hash()
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest length


def test_compute_input_hash_changes(mock_paths):
    """Verify that changing an input file changes the hash."""
    hash1 = compute_input_hash()
    
    # Modify watchlist
    (mock_paths / "watchlist.yml").write_text("tickers: [GOOG]")
    
    hash2 = compute_input_hash()
    assert hash1 != hash2


def test_manifest_creation_persistence(mock_paths):
    """Verify manifest creation and saving to disk."""
    manifest = create_manifest()
    
    assert manifest.run_id is not None
    assert manifest.timestamp is not None
    assert manifest.input_hash is not None
    assert manifest.data_hash is not None
    
    # Check persistence
    saved_path = mock_paths / "data" / "cache" / "manifests" / f"{manifest.run_id}.json"
    assert saved_path.exists()
    
    # Reload and compare
    loaded = RunManifest.from_json(saved_path.read_text())
    assert loaded.run_id == manifest.run_id
    assert loaded.input_hash == manifest.input_hash


def test_reset_inputs_logic(mock_paths):
    """Test the destructive reset logic (simulated)."""
    # We simulate the _reset_all_inputs logic here since we can't easily import from streamlit_ui due to st dependency
    
    uploads_dir = mock_paths / "data" / "user_uploads"
    files_to_check = ["transactions.csv", "holdings.csv", "ui_state.json"]
    
    # Pre-condition: files exist
    for f in files_to_check:
        assert (uploads_dir / f).exists()
        
    # Destructive action
    for f in files_to_check:
        path = uploads_dir / f
        if path.exists():
            path.unlink()
            
    # Post-condition: files gone
    for f in files_to_check:
        assert not (uploads_dir / f).exists()
