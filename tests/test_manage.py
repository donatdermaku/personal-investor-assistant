import pytest
import argparse
from unittest.mock import patch, MagicMock
from manage import cmd_compute, cmd_migrate, cmd_update

def test_cmd_compute():
    args = argparse.Namespace(portfolio="default")
    
    with patch("src.pipeline.compute_app_state") as mock_compute, \
         patch("src.pipeline.save_artifacts") as mock_save:
             
        # Mock return object with run_id
        mock_state = MagicMock()
        mock_state.run_manifest.run_id = "test-run"
        mock_compute.return_value = mock_state
        
        cmd_compute(args)
        
        mock_compute.assert_called_once()
        mock_save.assert_called_once()

def test_cmd_migrate():
    with patch("scripts.migrate_files_to_db.migrate") as mock_mig:
        cmd_migrate(None)
        mock_mig.assert_called_once()
