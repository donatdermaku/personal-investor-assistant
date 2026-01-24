from __future__ import annotations

import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def get_engine():
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not set.")
    return create_engine(SUPABASE_DB_URL, pool_pre_ping=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine() if SUPABASE_DB_URL else None)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
