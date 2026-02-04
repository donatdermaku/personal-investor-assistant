import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime
from src.coverage_meta import CoverageMeta
from src.app_state import AppState
from src.manifest import RunManifest
from src.streamlit_export import generate_html_report
from src.risk_free import RiskFreeSeries

@pytest.fixture
def mock_app_state():
    """Create a mock AppState with minimal required fields."""
    
    # Mock Portfolio Result
    mock_portfolio = MagicMock()
    mock_portfolio.twr = 0.1234
    mock_portfolio.mwr = 0.1050
    mock_portfolio.daily_values.empty = False
    mock_portfolio.errors = []
    
    # Mock Manifest
    mock_manifest = RunManifest(
        run_id="test-run-id-123",
        timestamp=datetime.now().isoformat(),
        input_hash="abc",
        data_hash="def",
        code_version="dev",
        coverage_summary={}
    )
    
    # Mock Meta
    meta = CoverageMeta(
        total=10, 
        covered=8, 
        last_date="2023-01-01", 
        missing_tickers=["BAD"],
        reasons={},
        notes=[]
    )
    
    return AppState(
        run_manifest=mock_manifest,
        portfolio=mock_portfolio,
        prices=MagicMock(),
        scores=MagicMock(),
        watch_tickers=["AAPL", "MSFT", "GOOG"],
        price_meta=meta,
        fundamentals_meta=meta,
        scores_meta=meta,
        benchmark_prices=MagicMock(),
        risk_free=RiskFreeSeries(series=pd.DataFrame(), status="unavailable", reason_codes=["MISSING_DTB3"]),
        market_state="Open"
    )

def test_generate_html_report_structure(mock_app_state):
    """Verify HTML report contains key sections and data."""
    html = generate_html_report(mock_app_state)
    
    # Check for HTML structure
    assert "<!doctype html>" in html
    assert "Investor Report" in html
    
    # Check for Metadata
    assert "Run ID: test-run-id-123" in html
    
    # Check for Metrics
    assert "12.34%" in html  # TWR
    assert "10.50%" in html  # MWR
    assert "8/10" in html     # Coverage
    
    # Check for Holdings
    assert "AAPL" in html
    assert "MSFT" in html
    
    # Check for Sections
    assert "Executive Summary" in html
    assert "Glossary" in html

def test_generate_html_report_no_manifest(mock_app_state):
    """Verify report generation handles missing manifest gracefully."""
    mock_app_state.run_manifest = None
    html = generate_html_report(mock_app_state)
    
    assert "Run ID: N/A" in html
