"""
Portfolio Repository

Handles database operations for Portfolio entities.
"""

from typing import Optional
from storage.db import session_scope
from storage.models import Portfolio


class PortfolioRepository:
    """Repository for portfolio database operations."""
    
    def get_by_id(self, portfolio_id: int) -> Optional[Portfolio]:
        """
        Get portfolio by ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio instance or None if not found
        """
        with session_scope() as session:
            return session.query(Portfolio).filter_by(id=portfolio_id).first()
    
    def get_or_create_default(self, user_id: int, name: str = "Default Portfolio") -> Portfolio:
        """
        Get or create default portfolio for user.
        
        Args:
            user_id: User ID
            name: Portfolio name
            
        Returns:
            Portfolio instance
        """
        with session_scope() as session:
            portfolio = session.query(Portfolio).filter_by(
                user_id=user_id,
                name=name
            ).first()
            
            if not portfolio:
                portfolio = Portfolio(user_id=user_id, name=name)
                session.add(portfolio)
                session.commit()
                session.refresh(portfolio)
            
            return portfolio
    
    def exists(self, portfolio_id: int) -> bool:
        """
        Check if portfolio exists.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            True if exists, False otherwise
        """
        with session_scope() as session:
            return session.query(Portfolio).filter_by(id=portfolio_id).count() > 0
