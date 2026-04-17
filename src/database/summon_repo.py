"""
Repository for summon_analysis.db.

Stores every observed summon and merge event from the player's board.
Used by SummonAnalyzer to test whether the 20% uniform distribution holds
and whether trigger context (manual vs post-merge) affects the distribution.

Tables
------
summon_sessions  — one row per tracked match (deck confirmed at match end)
summon_events    — one row per rank-1 unit appearing in an empty cell
merge_events     — one row per detected merge on the player's board
"""

import sqlite3
from datetime import datetime, timezone


class SummonRepo:

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @staticmethod
    def open_session(conn: sqlite3.Connection, match_id: str):
        """Insert a session row at match start (deck_json filled later)."""
        conn.execute(
            """INSERT OR IGNORE INTO summon_sessions
               (match_id, recorded_at, total_summons, total_merges)
               VALUES (?, ?, 0, 0)""",
            (match_id, datetime.now(timezone.utc).isoformat()),
        )

    @staticmethod
    def close_session(conn: sqlite3.Connection, match_id: str,
                      deck_json: str, total_summons: int, total_merges: int):
        """Finalise the session with the confirmed deck and event counts."""
        conn.execute(
            """UPDATE summon_sessions
               SET deck_json = ?, total_summons = ?, total_merges = ?
               WHERE match_id = ?""",
            (deck_json, total_summons, total_merges, match_id),
        )

    # ------------------------------------------------------------------
    # Summon events
    # ------------------------------------------------------------------

    @staticmethod
    def insert_summon(conn: sqlite3.Connection, data: dict):
        """
        Insert one summon event.

        Expected keys: match_id, timestamp_sec, wave_number,
                       unit_summoned, trigger_type,
                       merged_unit_id (opt), merged_from_rank (opt)
        """
        data.setdefault("merged_unit_id", None)
        data.setdefault("merged_from_rank", None)
        conn.execute(
            """INSERT INTO summon_events
               (match_id, timestamp_sec, wave_number, unit_summoned,
                trigger_type, merged_unit_id, merged_from_rank)
               VALUES (:match_id, :timestamp_sec, :wave_number, :unit_summoned,
                       :trigger_type, :merged_unit_id, :merged_from_rank)""",
            data,
        )

    @staticmethod
    def get_summons_for_session(conn: sqlite3.Connection,
                                match_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM summon_events WHERE match_id = ? ORDER BY timestamp_sec",
            (match_id,),
        ).fetchall()

    @staticmethod
    def get_all_summons(conn: sqlite3.Connection,
                        trigger_type: str | None = None) -> list[sqlite3.Row]:
        """
        Return all summon events, optionally filtered by trigger_type.
        Only includes events from sessions with a confirmed deck_json.
        """
        if trigger_type:
            return conn.execute(
                """SELECT e.*, s.deck_json
                   FROM summon_events e
                   JOIN summon_sessions s USING (match_id)
                   WHERE s.deck_json IS NOT NULL
                     AND e.trigger_type = ?
                   ORDER BY e.id""",
                (trigger_type,),
            ).fetchall()
        return conn.execute(
            """SELECT e.*, s.deck_json
               FROM summon_events e
               JOIN summon_sessions s USING (match_id)
               WHERE s.deck_json IS NOT NULL
               ORDER BY e.id""",
        ).fetchall()

    # ------------------------------------------------------------------
    # Merge events
    # ------------------------------------------------------------------

    @staticmethod
    def insert_merge(conn: sqlite3.Connection, data: dict):
        """
        Insert one merge event.

        Expected keys: match_id, timestamp_sec, wave_number,
                       unit_id, from_rank, to_rank
        """
        conn.execute(
            """INSERT INTO merge_events
               (match_id, timestamp_sec, wave_number, unit_id, from_rank, to_rank)
               VALUES (:match_id, :timestamp_sec, :wave_number,
                       :unit_id, :from_rank, :to_rank)""",
            data,
        )

    @staticmethod
    def get_merges_for_session(conn: sqlite3.Connection,
                               match_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM merge_events WHERE match_id = ? ORDER BY timestamp_sec",
            (match_id,),
        ).fetchall()

    # ------------------------------------------------------------------
    # Summary query for the UI
    # ------------------------------------------------------------------

    @staticmethod
    def get_unit_summon_counts(
            conn: sqlite3.Connection,
            trigger_type: str | None = None,
    ) -> list[sqlite3.Row]:
        """
        Return (unit_summoned, count) aggregated across all sessions that
        have a confirmed deck.  Optionally filter by trigger_type.
        """
        if trigger_type:
            return conn.execute(
                """SELECT unit_summoned, COUNT(*) AS count
                   FROM summon_events e
                   JOIN summon_sessions s USING (match_id)
                   WHERE s.deck_json IS NOT NULL
                     AND e.trigger_type = ?
                   GROUP BY unit_summoned
                   ORDER BY count DESC""",
                (trigger_type,),
            ).fetchall()
        return conn.execute(
            """SELECT unit_summoned, COUNT(*) AS count
               FROM summon_events e
               JOIN summon_sessions s USING (match_id)
               WHERE s.deck_json IS NOT NULL
               GROUP BY unit_summoned
               ORDER BY count DESC""",
        ).fetchall()

    @staticmethod
    def get_unit_merge_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Return (unit_id, from_rank, count) for all confirmed sessions."""
        return conn.execute(
            """SELECT m.unit_id, m.from_rank, COUNT(*) AS count
               FROM merge_events m
               JOIN summon_sessions s USING (match_id)
               WHERE s.deck_json IS NOT NULL
               GROUP BY m.unit_id, m.from_rank
               ORDER BY m.unit_id, m.from_rank""",
        ).fetchall()

    @staticmethod
    def get_session_count(conn: sqlite3.Connection) -> int:
        """How many completed (deck-confirmed) sessions are in the DB."""
        row = conn.execute(
            "SELECT COUNT(*) FROM summon_sessions WHERE deck_json IS NOT NULL"
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def get_total_summon_count(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """SELECT COALESCE(SUM(total_summons), 0)
               FROM summon_sessions WHERE deck_json IS NOT NULL"""
        ).fetchone()
        return row[0] if row else 0
