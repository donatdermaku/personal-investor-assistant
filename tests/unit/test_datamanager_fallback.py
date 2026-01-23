import pytest
import yaml
from unittest.mock import patch, MagicMock
from storage.datamanager import DataManager, data_manager
from storage import repo, db

@pytest.fixture
def mock_root(tmp_path):
    (tmp_path / "watchlist.yml").write_text("tickers: ['FILE_ONLY']")
    return tmp_path

def test_fallback_logic(mock_root, tmp_path):
    db_path = tmp_path / "fallback_test.db"
    
    # 1. Hybrid Mode (Expected default behavior)
    # Mocking environment and paths
    with patch("storage.datamanager.STORAGE_MODE", "hybrid"), \
         patch("storage.repo.list_watch_tickers", return_value=[]), \
         patch("storage.datamanager.ROOT", mock_root):
        
        dm = DataManager()
        # Should fallback to file because DB gives empty list
        assert dm.load_watchlist() == ["FILE_ONLY"]

    # 2. DB Mode (Strict)
    with patch("storage.datamanager.STORAGE_MODE", "db"), \
         patch("storage.repo.list_watch_tickers", return_value=["DB_ONLY"]), \
         patch("storage.datamanager.ROOT", mock_root):
        
        dm = DataManager()
        # Should ignore file and return DB data
        assert dm.load_watchlist() == ["DB_ONLY"]

    # 3. DB Mode (Strict) with Empty DB
    with patch("storage.datamanager.STORAGE_MODE", "db"), \
         patch("storage.repo.list_watch_tickers", return_value=[]), \
         patch("storage.datamanager.ROOT", mock_root):
        
        dm = DataManager()
        # Should return empty list, NOT fallback
        assert dm.load_watchlist() == []

def test_save_propagation(mock_root, tmp_path):
    # Verify saves go to both in hybrid
    with patch("storage.datamanager.STORAGE_MODE", "hybrid"), \
         patch("storage.repo.replace_watchlist") as mock_db_save, \
         patch("storage.datamanager.ROOT", mock_root):
        
        dm = DataManager()
        dm.save_watchlist(["NEW_TICKER"])
        
        # DB should be called
        assert mock_db_save.called
        
        # File should be written
        content = (mock_root / "watchlist.yml").read_text()
        assert "NEW_TICKER" in content
