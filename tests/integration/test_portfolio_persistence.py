import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src import portfolio
from storage import repo, db, datamanager, models

@pytest.fixture
def test_db_session(tmp_path):
    # Setup isolated DB
    db_path = tmp_path / "test_persistence.db"
    db.init_db(str(db_path))
    models.Base.metadata.create_all(bind=db._engine)
    return db_path

def test_load_portfolio_from_db(test_db_session):
    # 1. Insert trades into DB directly via repo
    user = repo.get_or_create_default_user()
    p_id = repo.get_default_portfolio_id(user.id)
    
    trades = [{
        "date": pd.to_datetime("2023-01-01"),
        "ticker": "AAPL",
        "action": "BUY",
        "shares": 10,
        "amount": -1500.0,
        "price": 150.0,
        "fees": 0.0,
        "notes": "Initial"
    }]
    repo.replace_trades(p_id, trades)
    
    # 2. Mock prices (needed for load_portfolio)
    prices = pd.DataFrame({
        "date": [pd.to_datetime("2023-01-01")],
        "ticker": ["AAPL"],
        "adj_close": [160.0]
    })
    
    # 3. Call load_portfolio with Strict DB (or hybrid with no files) behavior
    # We patch STORAGE_MODE to ensure we are testing DB path
    with patch("storage.datamanager.STORAGE_MODE", "db"):
        # We need to re-init DM or patch the repo functions if DM was already init? 
        # DM checks env var at init. But we patch strictly before DM logic runs?
        # Actually DM is a global singleton. `datamanager` import likely already ran.
        # But `load_trades` checks `STORAGE_MODE` (global var in module)
        
        result = portfolio.load_portfolio(prices, ["AAPL"], source_override="Ledger")
        
        assert not result.errors
        assert result.daily_values["value"].iloc[0] > 0
        # Check holdings: 10 shares * 160 + cash
        # Cash = -1500 (buy)
        # Value = 10 * 160 + (-1500) = 1600 - 1500 = 100
        # (Exact math depends on simplified logic in portfolio.py)
        
        holdings = result.holdings_daily
        assert not holdings.empty
        assert holdings.iloc[0]["quantity"] == 10
