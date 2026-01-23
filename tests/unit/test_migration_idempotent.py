import pytest
import os
import yaml
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts import migrate_files_to_db
from storage import repo, db, models

@pytest.fixture
def mock_root(tmp_path):
    """Mock ROOT path to point to a temp dir with sample files."""
    # Setup sample files
    (tmp_path / "watchlist.yml").write_text("tickers: ['AAPL', 'GOOG']")
    
    data_dir = tmp_path / "data" / "user_uploads"
    data_dir.mkdir(parents=True)
    
    # Sample Transactions
    pd.DataFrame({
        "date": ["2023-01-01"],
        "ticker": ["AAPL"],
        "action": ["BUY"],
        "shares": [10],
        "amount": [-1500]
    }).to_csv(data_dir / "transactions.csv", index=False)
    
    # Sample Holdings
    pd.DataFrame({
        "ticker": ["GOOG"],
        "shares": [5]
    }).to_csv(data_dir / "holdings.csv", index=False)
    
    return tmp_path

def test_migration_idempotency(mock_root, tmp_path):
    # Override ROOT and DB path in migration script context
    db_path = tmp_path / "migration_test.db"
    
    # Mocking storage.db to use our test DB
    with patch("storage.db.DB_PATH", str(db_path)), \
         patch("scripts.migrate_files_to_db.ROOT", mock_root), \
         patch("storage.datamanager.ROOT", mock_root):
        
        # Reset engine just in case it was init before
        db._engine = None 
        
        # Run migration #1
        migrate_files_to_db.migrate()
        
        # Verify counts
        user = repo.get_or_create_default_user()
        p_id = repo.get_default_portfolio_id(user.id)
        
        assert len(repo.list_watch_tickers(user.id)) == 2
        assert len(repo.get_trades(p_id)) == 1
        assert len(repo.get_latest_snapshot(p_id)) == 1
        
        # Run migration #2 (Should verify idempotency)
        migrate_files_to_db.migrate()
        
        # Counts should remain same (replace logic)
        assert len(repo.list_watch_tickers(user.id)) == 2
        assert len(repo.get_trades(p_id)) == 1
        assert len(repo.get_latest_snapshot(p_id)) == 1
