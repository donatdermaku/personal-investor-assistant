"""
Run Repository

Handles database operations for Run entities (portfolio computation runs).
"""

from typing import Optional, List
from datetime import datetime, timezone
from storage.db import session_scope
from storage.models import Run


class RunRepository:
    """Repository for run database operations."""
    
    def get_by_id(self, run_id: str) -> Optional[Run]:
        """
        Get run by ID.
        
        Args:
            run_id: Run ID (UUID)
            
        Returns:
            Run instance or None if not found
        """
        with session_scope() as session:
            return session.query(Run).filter_by(run_id=run_id).first()
    
    def get_latest(self) -> Optional[Run]:
        """
        Get the most recent completed run.
        
        Returns:
            Latest Run instance or None
        """
        with session_scope() as session:
            return (
                session.query(Run)
                .filter_by(status="completed")
                .order_by(Run.completed_at.desc())
                .first()
            )
    
    def list_completed(self, limit: int = 100) -> List[Run]:
        """
        List completed runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of completed Run instances
        """
        with session_scope() as session:
            return (
                session.query(Run)
                .filter_by(status="completed")
                .order_by(Run.completed_at.desc())
                .limit(limit)
                .all()
            )
    
    def create(
        self,
        run_id: str,
        portfolio_id: int,
        input_hash: str,
        data_hash: str,
        manifest_json: str | None = None
    ) -> Run:
        """
        Create a new run record.
        
        Args:
            run_id: Run ID (UUID)
            portfolio_id: Portfolio ID
            input_hash: Hash of input data
            data_hash: Hash of market data
            manifest_json: Optional JSON manifest
            
        Returns:
            Created Run instance
        """
        with session_scope() as session:
            run = Run(
                run_id=run_id,
                portfolio_id=portfolio_id,
                status="in_progress",
                input_hash=input_hash,
                data_hash=data_hash,
                manifest_json=manifest_json,
                created_at=datetime.now(timezone.utc),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run
    
    def mark_completed(self, run_id: str) -> None:
        """
        Mark a run as completed.
        
        Args:
            run_id: Run ID to mark as completed
        """
        with session_scope() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                session.commit()
    
    def mark_failed(self, run_id: str, error_code: str, error_message: str) -> None:
        """
        Mark a run as failed.
        
        Args:
            run_id: Run ID to mark as failed
            error_code: Error code
            error_message: Error message
        """
        with session_scope() as session:
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                run.status = "failed"
                run.error_code = error_code
                run.error_message = error_message
                run.completed_at = datetime.now(timezone.utc)
                session.commit()
