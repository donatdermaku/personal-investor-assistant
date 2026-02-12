import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_api_rate_limiter():
    """Prevent cross-test pollution from the in-memory API rate limiter."""
    try:
        import src.api.server as server

        server._rate_limiter._requests.clear()
    except Exception:
        # Keep fixture non-invasive for tests that don't import API server.
        pass
    yield
