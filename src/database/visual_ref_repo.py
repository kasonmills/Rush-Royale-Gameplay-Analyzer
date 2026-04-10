"""
Repository for visual_reference.db.

Tracks which reference images have been captured for each unit variant,
talent icon, and hero portrait.

Example:
    from src.database.connection import visual_ref_db
    from src.database.visual_ref_repo import VisualRefRepo, TalentIconRefRepo, HeroPortraitRefRepo

    with visual_ref_db() as conn:
        missing = VisualRefRepo.get_uncaptured(conn, "inquisitor")
"""

import sqlite3


class VisualRefRepo:
    """Unit sprite reference images (body art per rank/appearance state)."""

    @staticmethod
    def get(conn: sqlite3.Connection, unit_id: str, appearance_state: str,
            merge_rank: int, variant_tag: str | None = None) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM visual_reference "
            "WHERE unit_id = ? AND appearance_state = ? AND merge_rank = ? AND variant_tag IS ?",
            (unit_id, appearance_state, merge_rank, variant_tag)
        ).fetchone()

    @staticmethod
    def get_all_for_unit(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM visual_reference WHERE unit_id = ? "
            "ORDER BY appearance_state, merge_rank",
            (unit_id,)
        ).fetchall()

    @staticmethod
    def get_captured(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM visual_reference WHERE unit_id = ? AND captured = 1 "
            "ORDER BY appearance_state, merge_rank",
            (unit_id,)
        ).fetchall()

    @staticmethod
    def get_uncaptured(conn: sqlite3.Connection, unit_id: str | None = None) -> list[sqlite3.Row]:
        if unit_id:
            return conn.execute(
                "SELECT * FROM visual_reference WHERE unit_id = ? AND captured = 0 "
                "ORDER BY unit_id, appearance_state, merge_rank",
                (unit_id,)
            ).fetchall()
        return conn.execute(
            "SELECT * FROM visual_reference WHERE captured = 0 "
            "ORDER BY unit_id, appearance_state, merge_rank"
        ).fetchall()

    @staticmethod
    def mark_captured(conn: sqlite3.Connection, unit_id: str, appearance_state: str,
                      merge_rank: int, file_path: str, game_version: str,
                      variant_tag: str | None = None):
        conn.execute(
            "UPDATE visual_reference SET captured = 1, file_path = ?, game_version = ? "
            "WHERE unit_id = ? AND appearance_state = ? AND merge_rank = ? AND variant_tag IS ?",
            (file_path, game_version, unit_id, appearance_state, merge_rank, variant_tag)
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for data in rows:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO visual_reference ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(unit_id, appearance_state, merge_rank, variant_tag) DO UPDATE SET "
                f"file_path = excluded.file_path, "
                f"game_version = excluded.game_version, "
                f"captured = excluded.captured",
                list(data.values())
            )

    @staticmethod
    def capture_progress(conn: sqlite3.Connection) -> dict:
        """Returns a summary dict: {unit_id: {'captured': n, 'total': n}}."""
        rows = conn.execute(
            "SELECT unit_id, captured, COUNT(*) as cnt "
            "FROM visual_reference GROUP BY unit_id, captured"
        ).fetchall()
        progress = {}
        for row in rows:
            uid = row["unit_id"]
            if uid not in progress:
                progress[uid] = {"captured": 0, "total": 0}
            progress[uid]["total"] += row["cnt"]
            if row["captured"]:
                progress[uid]["captured"] += row["cnt"]
        return progress


class TalentIconRefRepo:
    """Talent tier badge reference images."""

    @staticmethod
    def get_uncaptured(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM talent_icon_reference WHERE captured = 0 ORDER BY unit_id, tier"
        ).fetchall()

    @staticmethod
    def mark_captured(conn: sqlite3.Connection, unit_id: str, tier: int,
                      file_path: str, game_version: str):
        conn.execute(
            "UPDATE talent_icon_reference SET captured = 1, file_path = ?, game_version = ? "
            "WHERE unit_id = ? AND tier = ?",
            (file_path, game_version, unit_id, tier)
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for data in rows:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO talent_icon_reference ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(unit_id, tier) DO UPDATE SET "
                f"file_path = excluded.file_path, "
                f"game_version = excluded.game_version, "
                f"captured = excluded.captured",
                list(data.values())
            )


class HeroPortraitRefRepo:
    """Hero portrait reference images."""

    @staticmethod
    def get_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute("SELECT * FROM hero_portrait_reference ORDER BY hero_id").fetchall()

    @staticmethod
    def get_uncaptured(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM hero_portrait_reference WHERE captured = 0 ORDER BY hero_id"
        ).fetchall()

    @staticmethod
    def mark_captured(conn: sqlite3.Connection, hero_id: str, file_path: str, game_version: str):
        conn.execute(
            "UPDATE hero_portrait_reference SET captured = 1, file_path = ?, game_version = ? "
            "WHERE hero_id = ?",
            (file_path, game_version, hero_id)
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for data in rows:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO hero_portrait_reference ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(hero_id) DO UPDATE SET "
                f"file_path = excluded.file_path, "
                f"game_version = excluded.game_version, "
                f"captured = excluded.captured",
                list(data.values())
            )