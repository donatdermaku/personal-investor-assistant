from __future__ import annotations

import logging
import os
import resource
import sys

logger = logging.getLogger(__name__)


def _rss_bytes() -> int | None:
    """Best-effort RSS in bytes (current if available)."""
    # Prefer /proc on Linux for current RSS.
    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r", encoding="utf-8") as handle:
                parts = handle.read().strip().split()
            if len(parts) >= 2:
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return rss_pages * page_size
    except Exception:
        pass

    # Fallback: ru_maxrss (platform-specific units).
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)  # bytes on macOS
        return int(rss) * 1024  # kilobytes on Linux
    except Exception:
        return None


def log_rss(stage: str) -> None:
    rss = _rss_bytes()
    if rss is None:
        logger.info("RSS %s: unknown", stage)
        return
    logger.info("RSS %s: %.2f MB", stage, rss / (1024 * 1024))
