"""
Repository for match_history.db.

Stores logged game state timelines with known outcomes — the training data
for the Phase 2 LightGBM win probability classifier.

Example:
    from src.database.connection import match_history_db
    from src.database.match_history_repo import MatchRepo, SnapshotRepo

    with match_history_db() as conn:
        MatchRepo.insert(conn, {...})
        SnapshotRepo.insert_many(conn, snapshots)
"""

import sqlite3


class MatchRepo:

    @staticmethod
    def insert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        conn.execute(
            f"INSERT OR IGNORE INTO matches ({cols}) VALUES ({placeholders})",
            list(data.values())
        )

    @staticmethod
    def get(conn: sqlite3.Connection, match_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()

    @staticmethod
    def get_recent(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM matches ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()

    @staticmethod
    def set_outcome(conn: sqlite3.Connection, match_id: str, outcome: str):
        conn.execute(
            "UPDATE matches SET outcome = ? WHERE match_id = ?", (outcome, match_id)
        )

    @staticmethod
    def get_labeled(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Returns all matches with a known outcome — used for LightGBM training."""
        return conn.execute(
            "SELECT * FROM matches WHERE outcome IS NOT NULL ORDER BY recorded_at"
        ).fetchall()


class SnapshotRepo:

    @staticmethod
    def insert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        conn.execute(
            f"INSERT INTO game_state_snapshots ({cols}) VALUES ({placeholders})",
            list(data.values())
        )

    @staticmethod
    def insert_many(conn: sqlite3.Connection, rows: list[dict]):
        for row in rows:
            SnapshotRepo.insert(conn, row)

    @staticmethod
    def get_for_match(conn: sqlite3.Connection, match_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM game_state_snapshots WHERE match_id = ? ORDER BY timestamp_sec",
            (match_id,)
        ).fetchall()


class UnitPerformanceRepo:

    @staticmethod
    def insert_many(conn: sqlite3.Connection, rows: list[dict]):
        for data in rows:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO unit_performance ({cols}) VALUES ({placeholders})",
                list(data.values())
            )

    @staticmethod
    def get_for_match(conn: sqlite3.Connection, match_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM unit_performance WHERE match_id = ? ORDER BY unit_id",
            (match_id,)
        ).fetchall()