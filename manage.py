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

def _stage_ingest_universe(args) -> None:
    from src.ingest.universe import main as ingest_universe_main
    ingest_universe_main()


def _stage_ingest_prices(args) -> None:
    from src.ingest.prices import main as ingest_prices_main
    ingest_prices_main()


def _stage_ingest_fundamentals(args) -> None:
    from src.ingest.fundamentals_sec import main as ingest_fundamentals_main
    ingest_fundamentals_main()


def _stage_compute_factors(args) -> None:
    from src.compute.factors import main as compute_factors_main
    compute_factors_main()


def _stage_compute_analytics(args) -> None:
    cmd_compute(args)


def _resolve_update_stages(args):
    compute_stage = [("compute_analytics", _stage_compute_analytics)]
    if getattr(args, "compute_only", False):
        return compute_stage
    return [
        ("ingest_universe", _stage_ingest_universe),
        ("ingest_prices", _stage_ingest_prices),
        ("ingest_fundamentals", _stage_ingest_fundamentals),
        ("compute_factors", _stage_compute_factors),
        *compute_stage,
    ]


def cmd_update(args):
    """Run staged update workflow with optional compute-only and dry-run modes."""
    logger.info("Running update sequence...")
    stages = _resolve_update_stages(args)
    stage_names = [name for name, _ in stages]

    if getattr(args, "dry_run", False):
        logger.info("Dry run enabled. Planned stages: %s", " -> ".join(stage_names))
        print("\n".join(stage_names))
        return

    for stage_name, stage_fn in stages:
        logger.info("Starting stage: %s", stage_name)
        try:
            stage_fn(args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code != 0:
                logger.error("Stage failed: %s (exit=%s)", stage_name, code)
                raise SystemExit(code)
        except Exception as exc:
            logger.error("Stage failed: %s (%s)", stage_name, exc, exc_info=True)
            raise SystemExit(1)
        logger.info("Completed stage: %s", stage_name)

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
    p_update.add_argument("--compute-only", action="store_true", help="Skip ingest stages and run compute only.")
    p_update.add_argument("--dry-run", action="store_true", help="Print stages without executing them.")
    p_update.add_argument("--portfolio", default="default", help="Portfolio alias (default: default)")
    p_update.set_defaults(func=cmd_update)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
