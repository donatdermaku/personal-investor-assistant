import os
import pytest

from storage import repo


def test_supabase_backend_optional():
    if not (os.getenv("SUPABASE_DB_URL") and os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        pytest.skip("Supabase env vars not set")
    assert repo.use_supabase()
