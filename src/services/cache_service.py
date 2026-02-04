"""
Cache Service

Centralized cache monitoring and management.
"""

from typing import List, Dict, Any
import logging
from dataclasses import asdict

from storage.cache_index import get_cache_entry
from market_data.persistent_cache import get_cache_age, clear_stale_cache
from storage.repo import use_supabase

logger = logging.getLogger(__name__)


class CacheService:
    """Service for cache monitoring and management."""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        if use_supabase():
            from storage_supabase.db import session_scope
            from storage_supabase import models
        else:
            from storage.db import session_scope
            from storage import models
        
        with session_scope() as session:
            all_entries = session.query(models.DataCacheIndex).all()
            
            total = len(all_entries)
            fresh_count = sum(1 for e in all_entries if e.status == "fresh")
            stale_count = sum(1 for e in all_entries if e.status == "stale")
            error_count = sum(1 for e in all_entries if e.status == "error")
            
            ages = []
            for entry in all_entries:
                age = get_cache_age(entry.source, entry.key)
                if age is not None:
                    ages.append(age)
            
            avg_age_hours = sum(ages) / len(ages) / 3600 if ages else 0
            
            return {
                "total_entries": total,
                "fresh_count": fresh_count,
                "stale_count": stale_count,
                "error_count": error_count,
                "avg_age_hours": round(avg_age_hours, 2),
            }
    
    def get_stale_caches(self) -> List[Dict[str, Any]]:
        """Return list of stale cache entries."""
        if use_supabase():
            from storage_supabase.db import session_scope
            from storage_supabase import models
        else:
            from storage.db import session_scope
            from storage import models
        
        with session_scope() as session:
            stale_entries = (
                session.query(models.DataCacheIndex)
                .filter_by(status="stale")
                .all()
            )
            
            return [
                {
                    "source": e.source,
                    "key": e.key,
                    "updated_at": e.updated_at.isoformat() if e.updated_at else None,
                    "error_code": e.error_code,
                }
                for e in stale_entries
            ]
    
    def clear_all_stale_caches(self) -> int:
        """Clear all stale caches, return count."""
        stale = self.get_stale_caches()
        cleared = 0
        
        for entry in stale:
            try:
                if clear_stale_cache(entry["source"], entry["key"]):
                    cleared += 1
                    logger.info(f"Cleared stale cache: {entry['source']}/{entry['key']}")
            except Exception as exc:
                logger.error(f"Failed to clear {entry['source']}/{entry['key']}: {exc}")
        
        return cleared
    
    def warm_cache_for_tickers(self, tickers: List[str], start: str, end: str) -> None:
        """Pre-fetch and cache data for given tickers."""
        from market_data.store import MarketDataStore
        
        store = MarketDataStore.default()
        for ticker in tickers:
            try:
                store.get_prices(ticker, start, end)
                logger.info(f"Warmed cache for {ticker}")
            except Exception as exc:
                logger.warning(f"Failed to warm cache for {ticker}: {exc}")
