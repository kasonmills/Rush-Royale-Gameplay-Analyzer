"""
Initialize all three SQLite databases.

Usage:
    python -m src.database.init_db
"""

import sqlite3
from pathlib import Path
from .schema import ALL_DDL

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def init_all():
    DATA_DIR.mkdir(exist_ok=True)
    for db_name, ddl in ALL_DDL.items():
        db_path = DATA_DIR / f"{db_name}.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(ddl)
        conn.commit()
        conn.close()
        print(f"Initialized {db_path}")


if __name__ == "__main__":
    init_all()
