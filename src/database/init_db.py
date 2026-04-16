"""
Initialize all three SQLite databases.

Usage:
    python -m src.database.init_db
"""

import sqlite3
from pathlib import Path
from .schema import ALL_DDL

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _migrate(conn: sqlite3.Connection):
    """
    Apply additive schema migrations for existing databases.
    Only ADD COLUMN statements here — never drop or rename.
    """
    migrations = [
        # stat_numbers: added talent_branch and talent_tier so each unit+talent
        # combination can have its own scaling curve / damage threshold.
        ("stat_numbers", "talent_branch", "ALTER TABLE stat_numbers ADD COLUMN talent_branch TEXT"),
        ("stat_numbers", "talent_tier",   "ALTER TABLE stat_numbers ADD COLUMN talent_tier INTEGER"),
        # stat_numbers: display_color removed — not needed for OCR or win prediction.
        # SQLite cannot DROP COLUMN on older versions; column is left in place but
        # never written to. It will be absent from any fresh database created from
        # the current DDL.
    ]
    for table, column, sql in migrations:
        if not _column_exists(conn, table, column):
            conn.execute(sql)


def init_all():
    DATA_DIR.mkdir(exist_ok=True)
    for db_name, ddl in ALL_DDL.items():
        db_path = DATA_DIR / f"{db_name}.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(ddl)
        if db_name == "unit_meta":
            _migrate(conn)
        conn.commit()
        conn.close()
        print(f"Initialized {db_path}")


if __name__ == "__main__":
    init_all()
