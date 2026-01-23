import pytest
import os
from storage import db, models, repo, datamanager

def test_imports():
    """Verify all storage modules import successfully."""
    assert db
    assert models
    assert repo
    assert datamanager

def test_db_init(tmp_path):
    """Verify DB engine creation and table creation."""
    db_path = tmp_path / "test.db"
    engine = db.init_db(str(db_path))
    assert engine
    
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    
    # Check if file exists
    assert db_path.exists()
