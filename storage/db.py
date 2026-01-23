import os
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Configurable path via env var, validation/default happens at import or init time
DB_PATH = os.getenv("USER_DB_PATH", "data/user.db")

def get_db_url(path: str = DB_PATH) -> str:
    # Ensure dir exists
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p}"

_engine = None
_SessionLocal = None

def init_db(path: str = DB_PATH):
    global _engine, _SessionLocal
    url = get_db_url(path)
    # check_same_thread=False is needed for SQLite when used across threads (e.g. Streamlit)
    _engine = create_engine(url, connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine

def get_engine():
    global _engine
    if _engine is None:
        init_db()
    return _engine

@contextmanager
def session_scope() -> Session:
    """Provide a transactional scope around a series of operations."""
    if _SessionLocal is None:
        init_db()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
