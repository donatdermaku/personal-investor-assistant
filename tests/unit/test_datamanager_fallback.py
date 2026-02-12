import pytest
from unittest.mock import patch
from storage.datamanager import DataManager

@pytest.fixture
def mock_root(tmp_path):
    (tmp_path / "watchlist.yml").write_text("tickers: ['FILE_ONLY']")
    return tmp_path

def test_mode_routing_for_watchlist(mock_root):
    # local mode reads from file
    with patch("storage.datamanager.STORAGE_MODE", "local"), \
         patch("storage.datamanager.ROOT", mock_root):
        dm = DataManager()
        assert dm.load_watchlist() == ["FILE_ONLY"]

    # supabase mode reads from repo only
    with patch("storage.datamanager.STORAGE_MODE", "supabase"), \
         patch("storage.repo.list_watch_tickers", return_value=["DB_ONLY"]), \
         patch("storage.datamanager.ROOT", mock_root):
        dm = DataManager()
        assert dm.load_watchlist() == ["DB_ONLY"]

    # supabase mode with empty repo stays empty (no file fallback)
    with patch("storage.datamanager.STORAGE_MODE", "supabase"), \
         patch("storage.repo.list_watch_tickers", return_value=[]), \
         patch("storage.datamanager.ROOT", mock_root):
        dm = DataManager()
        assert dm.load_watchlist() == []

def test_local_mode_saves_watchlist_to_file_only(mock_root):
    with patch("storage.datamanager.STORAGE_MODE", "local"), \
         patch("storage.repo.replace_watchlist") as mock_repo_save, \
         patch("storage.datamanager.ROOT", mock_root):
        dm = DataManager()
        dm.save_watchlist(["NEW_TICKER"])
        mock_repo_save.assert_not_called()
        content = (mock_root / "watchlist.yml").read_text()
        assert "NEW_TICKER" in content


def test_supabase_mode_saves_watchlist_to_repo_only(mock_root):
    with patch("storage.datamanager.STORAGE_MODE", "supabase"), \
         patch("storage.repo.replace_watchlist") as mock_repo_save, \
         patch("storage.datamanager.ROOT", mock_root):
        dm = DataManager()
        dm.save_watchlist(["NEW_TICKER"])
        mock_repo_save.assert_called_once()
        content = (mock_root / "watchlist.yml").read_text()
        assert "NEW_TICKER" not in content
