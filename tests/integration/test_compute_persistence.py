import pytest
import pandas as pd
from unittest.mock import patch
from src import pipeline
from storage import repo, db, models

@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test_compute.db"
    db.init_db(str(db_path))
    models.Base.metadata.create_all(bind=db._engine)
    return db_path

def test_compute_and_save(test_db, tmp_path):
    # Setup DataManager mode
    with patch("storage.datamanager.STORAGE_MODE", "supabase"), \
         patch("src.pipeline.EXPORTS_DIR", tmp_path / "exports"):
             
        # Create User/Portfolio
        user = repo.get_or_create_default_user()
        p_id = repo.get_default_portfolio_id(user.id)
        
        # Add some trades
        trades = [{
            "date": pd.to_datetime("2023-01-01"),
            "ticker": "AAPL",
            "action": "BUY",
            "shares": 10,
            "amount": -1000.0,
            "price": 100.0,
            "fees": 0.0,
            "notes": ""
        }]
        repo.replace_trades(p_id, trades)
        
        # Mock Market Data to avoid DuckDB/Network
        # We patch the getters in src.pipeline
        with patch("src.streamlit_data.get_prices") as mock_prices, \
             patch("src.streamlit_data.get_scores") as mock_scores, \
             patch("src.streamlit_data.get_fundamentals") as mock_fund:
            
            # Dummy Meta
            from src.streamlit_data import CoverageMeta
            dummy_meta = CoverageMeta(0, 0, [], None, {}, [])
                 
            # Return dummy market data with meta
            mock_prices.return_value = (
                pd.DataFrame({
                    "date": [pd.to_datetime("2023-01-01")], 
                    "ticker": ["AAPL"], 
                    "adj_close": [105.0]
                }), 
                dummy_meta
            )
            mock_scores.return_value = (pd.DataFrame(), dummy_meta)
            mock_fund.return_value = (pd.DataFrame(), dummy_meta)
            
            # --- ACTION: Compute ---
            app_state = pipeline.compute_app_state(portfolio_id=p_id, save_run=True)
            
            # Verify Run created
            with repo.session_scope() as session:
                runs = session.query(models.Run).all()
                assert len(runs) == 1
                assert runs[0].status == "completed"
                assert runs[0].manifest_json is not None
                run_id = runs[0].id
            
            # --- ACTION: Save Artifacts ---
            pipeline.save_artifacts(app_state)
            
            # Verify Artifacts created
            with repo.session_scope() as session:
                arts = session.query(models.Artifact).filter_by(run_id=run_id).all()
                # Should have summary, report, performance, monthly
                assert len(arts) >= 2 
            
            # Verify Files exist
            export_path = tmp_path / "exports" / run_id
            assert (export_path / "summary.json").exists()
            assert (export_path / "report.html").exists()
