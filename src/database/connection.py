"""
Database connection management for all three RRGA databases.

Usage:
    from src.database.connection import unit_meta_db, visual_ref_db, match_history_db

    with unit_meta_db() as conn:
        rows = conn.execute("SELECT * FROM units").fetchall()
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_DB_PATHS = {
    "unit_meta":        DATA_DIR / "unit_meta.db",
    "visual_reference": DATA_DIR / "visual_reference.db",
    "match_history":    DATA_DIR / "match_history.db",
}


def _connect(db_name: str) -> sqlite3.Connection:
    path = _DB_PATHS[db_name]
    if not path.exists():
        raise FileNotFoundError(
            f"{db_name}.db not found at {path}. "
            "Run: python -m src.database.init_db"
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row   # rows accessible by column name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def unit_meta_db():
    conn = _connect("unit_meta")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def visual_ref_db():
    conn = _connect("visual_reference")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def match_history_db():
    conn = _connect("match_history")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()