#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("manage")

def cmd_migrate(args):
    """Run data migration from files to SQLite."""
    from scripts.migrate_files_to_db import migrate
    logger.info("Starting migration...")
    migrate()
    logger.info("Migration finished.")

def cmd_compute(args):
    """Run headless computation and save artifacts."""
    from src.pipeline import compute_app_state, save_artifacts, EXPORTS_DIR
    
    logger.info("Starting headless computation...")
    # Trigger computation
    # P11.4 requirement: --portfolio default (we rely on default user logic for now)
    
    try:
        app_state = compute_app_state(save_run=True)
        save_artifacts(app_state)
        
        run_id = app_state.run_manifest.run_id
        logger.info(f"Computation complete. Run ID: {run_id}")
        logger.info(f"Artifacts saved to {EXPORTS_DIR}/{run_id}")
        
    except Exception as e:
        logger.error(f"Computation failed: {e}", exc_info=True)
        sys.exit(1)

def cmd_update(args):
    """Update market data (if supported) then compute."""
    # Future P12: Call 'ingest' pipeline here.
    # For now, it behaves like compute but ensures we run even if UI is closed.
    logger.info("Running update sequence...")
    cmd_compute(args)

def main():
    parser = argparse.ArgumentParser(description="Personal Investor Assistant Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Migrate
    p_migrate = subparsers.add_parser("migrate", help="Migrate legacy files to database")
    p_migrate.set_defaults(func=cmd_migrate)
    
    # Compute
    p_compute = subparsers.add_parser("compute", help="Run portfolio computation headlessly")
    p_compute.add_argument("--portfolio", default="default", help="Portfolio alias (default: default)")
    p_compute.set_defaults(func=cmd_compute)
    
    # Update
    p_update = subparsers.add_parser("update", help="Update market data and compute")
    p_update.set_defaults(func=cmd_update)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
