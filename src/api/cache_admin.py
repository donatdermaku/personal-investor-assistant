
# =====================================================
# Cache Management Admin Endpoints
# =====================================================

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    from src.services.cache_service import CacheService
    
    cache_service = CacheService()
    return cache_service.get_cache_stats()


@app.post("/admin/cache/clear-stale")
async def clear_stale_caches():
    """Clear all stale caches."""
    from src.services.cache_service import CacheService
    
    cache_service = CacheService()
    cleared = cache_service.clear_all_stale_caches()
    
    return {
        "cleared_count": cleared,
        "message": f"Cleared {cleared} stale cache entries"
    }


@app.get("/admin/cache/stale")
async def list_stale_caches():
    """List all stale cache entries."""
    from src.services.cache_service import CacheService
    
    cache_service = CacheService()
    return {
        "stale_caches": cache_service.get_stale_caches()
    }
