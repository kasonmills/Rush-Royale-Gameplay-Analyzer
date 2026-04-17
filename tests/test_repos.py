"""
Tests for the database repository layer.

Covers:
  MatchRepo      — insert, get, get_recent, set_outcome, get_labeled, purge_empty
  SnapshotRepo   — insert, insert_many, get_for_match
  UnitPerformanceRepo — insert_many, get_for_match
  SummonRepo     — open_session, close_session, insert_summon, insert_merge,
                   get_unit_summon_counts, get_unit_merge_counts,
                   get_session_count, get_total_summon_count

All tests use in-memory SQLite connections from conftest fixtures — no files
are written to disk and every test starts with a clean database.
"""

import json
from datetime import datetime, timezone

import pytest

from src.database.match_history_repo import MatchRepo, SnapshotRepo, UnitPerformanceRepo
from src.database.summon_repo import SummonRepo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _match_row(match_id: str = "m-001", **overrides) -> dict:
    base = dict(
        match_id=match_id,
        recorded_at=_now(),
        source_type="video_file",
        source_path=None,
        game_version="1.0",
        player_hero_id=None,
        opponent_hero_id=None,
        player_deck=None,
        opponent_deck=None,
        outcome=None,
        total_waves=None,
        match_duration_sec=None,
    )
    base.update(overrides)
    return base


def _snapshot_row(match_id: str = "m-001", ts: float = 0.0, **overrides) -> dict:
    base = dict(
        match_id=match_id,
        timestamp_sec=ts,
        wave_number=1,
        player_hp=3,
        opponent_hp=3,
        player_mana=0,
        player_board=json.dumps([]),
        opponent_board=json.dumps([]),
        active_buffs=json.dumps({}),
        win_probability=0.5,
        confidence=0.9,
    )
    base.update(overrides)
    return base


def _unit_perf_row(match_id: str = "m-001", unit_id: str = "archer",
                   player: str = "player", **overrides) -> dict:
    base = dict(
        match_id=match_id,
        unit_id=unit_id,
        player=player,
        max_rank_seen=3,
        talent_tier_seen=2,
        talent_branch="L",
        branch_confidence=0.85,
    )
    base.update(overrides)
    return base


# ===========================================================================
# MatchRepo
# ===========================================================================

class TestMatchRepo:

    def test_insert_and_get_round_trip(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-001"))
        row = MatchRepo.get(conn, "m-001")
        assert row is not None
        assert row["match_id"] == "m-001"

    def test_get_nonexistent_returns_none(self, match_history_conn):
        assert MatchRepo.get(match_history_conn, "no-such-match") is None

    def test_insert_or_ignore_does_not_raise_on_duplicate(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-dup"))
        MatchRepo.insert(conn, _match_row("m-dup"))  # should not raise
        # Only one row should exist
        rows = conn.execute("SELECT COUNT(*) FROM matches WHERE match_id='m-dup'").fetchone()
        assert rows[0] == 1

    def test_get_recent_returns_most_recent_first(self, match_history_conn):
        conn = match_history_conn
        for i in range(5):
            MatchRepo.insert(conn, _match_row(f"m-{i:03d}",
                                              recorded_at=f"2026-04-{i+1:02d}T00:00:00+00:00"))
        recent = MatchRepo.get_recent(conn, limit=3)
        assert len(recent) == 3
        # Should be descending by recorded_at
        times = [r["recorded_at"] for r in recent]
        assert times == sorted(times, reverse=True)

    def test_get_recent_limit_respected(self, match_history_conn):
        conn = match_history_conn
        for i in range(10):
            MatchRepo.insert(conn, _match_row(f"m-{i:03d}"))
        assert len(MatchRepo.get_recent(conn, limit=4)) == 4

    def test_set_outcome_updates_row(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-001"))
        MatchRepo.set_outcome(conn, "m-001", "win")
        row = MatchRepo.get(conn, "m-001")
        assert row["outcome"] == "win"

    def test_set_outcome_loss(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-002"))
        MatchRepo.set_outcome(conn, "m-002", "loss")
        assert MatchRepo.get(conn, "m-002")["outcome"] == "loss"

    def test_get_labeled_returns_only_matches_with_outcome(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-win"))
        MatchRepo.insert(conn, _match_row("m-loss"))
        MatchRepo.insert(conn, _match_row("m-none"))  # no outcome
        MatchRepo.set_outcome(conn, "m-win",  "win")
        MatchRepo.set_outcome(conn, "m-loss", "loss")
        labeled = MatchRepo.get_labeled(conn)
        ids = {r["match_id"] for r in labeled}
        assert "m-win"  in ids
        assert "m-loss" in ids
        assert "m-none" not in ids

    def test_purge_empty_removes_matches_with_no_snapshots(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-empty"))
        MatchRepo.insert(conn, _match_row("m-has-snap"))
        SnapshotRepo.insert(conn, _snapshot_row("m-has-snap"))
        deleted = MatchRepo.purge_empty(conn)
        assert deleted == 1
        assert MatchRepo.get(conn, "m-empty") is None
        assert MatchRepo.get(conn, "m-has-snap") is not None

    def test_purge_empty_returns_zero_when_nothing_to_purge(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-good"))
        SnapshotRepo.insert(conn, _snapshot_row("m-good"))
        assert MatchRepo.purge_empty(conn) == 0

    def test_purge_empty_on_empty_db_returns_zero(self, match_history_conn):
        assert MatchRepo.purge_empty(match_history_conn) == 0

    def test_source_type_stored_correctly(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-live", source_type="live_capture"))
        row = MatchRepo.get(conn, "m-live")
        assert row["source_type"] == "live_capture"


# ===========================================================================
# SnapshotRepo
# ===========================================================================

class TestSnapshotRepo:

    def test_insert_and_retrieve(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-snap"))
        SnapshotRepo.insert(conn, _snapshot_row("m-snap", ts=1.5, wave_number=3))
        rows = SnapshotRepo.get_for_match(conn, "m-snap")
        assert len(rows) == 1
        assert rows[0]["wave_number"] == 3
        assert rows[0]["timestamp_sec"] == pytest.approx(1.5)

    def test_get_for_match_returns_ordered_by_timestamp(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-order"))
        for ts in [5.0, 1.0, 3.0]:
            SnapshotRepo.insert(conn, _snapshot_row("m-order", ts=ts))
        rows = SnapshotRepo.get_for_match(conn, "m-order")
        times = [r["timestamp_sec"] for r in rows]
        assert times == sorted(times)

    def test_insert_many_inserts_all_rows(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-many"))
        rows = [_snapshot_row("m-many", ts=float(i)) for i in range(5)]
        SnapshotRepo.insert_many(conn, rows)
        result = SnapshotRepo.get_for_match(conn, "m-many")
        assert len(result) == 5

    def test_get_for_nonexistent_match_returns_empty(self, match_history_conn):
        result = SnapshotRepo.get_for_match(match_history_conn, "ghost-match")
        assert result == []

    def test_win_probability_stored_correctly(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-prob"))
        SnapshotRepo.insert(conn, _snapshot_row("m-prob", win_probability=0.73))
        row = SnapshotRepo.get_for_match(conn, "m-prob")[0]
        assert row["win_probability"] == pytest.approx(0.73)

    def test_hp_values_stored_correctly(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-hp"))
        SnapshotRepo.insert(conn, _snapshot_row("m-hp", player_hp=2, opponent_hp=1))
        row = SnapshotRepo.get_for_match(conn, "m-hp")[0]
        assert row["player_hp"] == 2
        assert row["opponent_hp"] == 1


# ===========================================================================
# UnitPerformanceRepo
# ===========================================================================

class TestUnitPerformanceRepo:

    def test_insert_many_and_retrieve(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-perf"))
        rows = [
            _unit_perf_row("m-perf", "archer", "player"),
            _unit_perf_row("m-perf", "knight", "player"),
            _unit_perf_row("m-perf", "rogue",  "opponent"),
        ]
        UnitPerformanceRepo.insert_many(conn, rows)
        result = UnitPerformanceRepo.get_for_match(conn, "m-perf")
        assert len(result) == 3

    def test_results_ordered_by_unit_id(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-ord"))
        UnitPerformanceRepo.insert_many(conn, [
            _unit_perf_row("m-ord", "rogue",   "player"),
            _unit_perf_row("m-ord", "archer",  "player"),
            _unit_perf_row("m-ord", "chemist", "player"),
        ])
        result = UnitPerformanceRepo.get_for_match(conn, "m-ord")
        ids = [r["unit_id"] for r in result]
        assert ids == sorted(ids)

    def test_talent_branch_stored_correctly(self, match_history_conn):
        conn = match_history_conn
        MatchRepo.insert(conn, _match_row("m-talent"))
        UnitPerformanceRepo.insert_many(conn, [
            _unit_perf_row("m-talent", talent_branch="R", branch_confidence=0.95)
        ])
        row = UnitPerformanceRepo.get_for_match(conn, "m-talent")[0]
        assert row["talent_branch"] == "R"
        assert row["branch_confidence"] == pytest.approx(0.95)

    def test_get_for_nonexistent_match_returns_empty(self, match_history_conn):
        result = UnitPerformanceRepo.get_for_match(match_history_conn, "ghost")
        assert result == []


# ===========================================================================
# SummonRepo
# ===========================================================================

class TestSummonRepo:

    def _open(self, conn, match_id="s-001", deck=None):
        SummonRepo.open_session(conn, match_id)
        if deck is not None:
            SummonRepo.close_session(conn, match_id, json.dumps(deck), 0, 0)

    def test_open_session_creates_row(self, summon_analysis_conn):
        conn = summon_analysis_conn
        SummonRepo.open_session(conn, "s-001")
        row = conn.execute(
            "SELECT * FROM summon_sessions WHERE match_id='s-001'"
        ).fetchone()
        assert row is not None

    def test_open_session_idempotent(self, summon_analysis_conn):
        conn = summon_analysis_conn
        SummonRepo.open_session(conn, "s-dup")
        SummonRepo.open_session(conn, "s-dup")  # should not raise
        count = conn.execute(
            "SELECT COUNT(*) FROM summon_sessions WHERE match_id='s-dup'"
        ).fetchone()[0]
        assert count == 1

    def test_close_session_sets_deck_and_counts(self, summon_analysis_conn):
        conn = summon_analysis_conn
        SummonRepo.open_session(conn, "s-close")
        deck = ["archer", "knight", "chemist"]
        SummonRepo.close_session(conn, "s-close", json.dumps(deck), 50, 10)
        row = conn.execute(
            "SELECT * FROM summon_sessions WHERE match_id='s-close'"
        ).fetchone()
        assert json.loads(row["deck_json"]) == deck
        assert row["total_summons"] == 50
        assert row["total_merges"] == 10

    def test_insert_summon_stores_event(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-ev", deck=["archer"])
        SummonRepo.insert_summon(conn, dict(
            match_id="s-ev",
            timestamp_sec=1.0,
            wave_number=2,
            unit_summoned="archer",
            trigger_type="manual",
        ))
        rows = SummonRepo.get_summons_for_session(conn, "s-ev")
        assert len(rows) == 1
        assert rows[0]["unit_summoned"] == "archer"
        assert rows[0]["trigger_type"] == "manual"

    def test_insert_summon_defaults_merged_fields_to_none(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-defaults", deck=["archer"])
        SummonRepo.insert_summon(conn, dict(
            match_id="s-defaults",
            timestamp_sec=0.0,
            wave_number=1,
            unit_summoned="archer",
            trigger_type="manual",
        ))
        row = SummonRepo.get_summons_for_session(conn, "s-defaults")[0]
        assert row["merged_unit_id"] is None
        assert row["merged_from_rank"] is None

    def test_get_all_summons_excludes_unconfirmed_sessions(self, summon_analysis_conn):
        conn = summon_analysis_conn
        # Open but don't close (no deck_json)
        SummonRepo.open_session(conn, "s-unconfirmed")
        SummonRepo.insert_summon(conn, dict(
            match_id="s-unconfirmed",
            timestamp_sec=0.0,
            wave_number=1,
            unit_summoned="archer",
            trigger_type="manual",
        ))
        result = SummonRepo.get_all_summons(conn)
        assert len(result) == 0

    def test_get_all_summons_includes_confirmed_sessions(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-confirmed", deck=["archer"])
        SummonRepo.insert_summon(conn, dict(
            match_id="s-confirmed",
            timestamp_sec=0.0,
            wave_number=1,
            unit_summoned="archer",
            trigger_type="manual",
        ))
        result = SummonRepo.get_all_summons(conn)
        assert len(result) == 1

    def test_get_all_summons_filter_by_trigger_type(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-trig", deck=["archer", "knight"])
        for tt in ("manual", "post_merge", "manual"):
            SummonRepo.insert_summon(conn, dict(
                match_id="s-trig",
                timestamp_sec=0.0,
                wave_number=1,
                unit_summoned="archer",
                trigger_type=tt,
            ))
        manual = SummonRepo.get_all_summons(conn, trigger_type="manual")
        post = SummonRepo.get_all_summons(conn, trigger_type="post_merge")
        assert len(manual) == 2
        assert len(post) == 1

    def test_insert_merge_stores_event(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-merge", deck=["archer"])
        SummonRepo.insert_merge(conn, dict(
            match_id="s-merge",
            timestamp_sec=2.0,
            wave_number=3,
            unit_id="archer",
            from_rank=1,
            to_rank=2,
        ))
        rows = SummonRepo.get_merges_for_session(conn, "s-merge")
        assert len(rows) == 1
        assert rows[0]["unit_id"] == "archer"
        assert rows[0]["from_rank"] == 1
        assert rows[0]["to_rank"] == 2

    def test_get_unit_summon_counts_aggregates_correctly(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-counts", deck=["archer", "knight"])
        for _ in range(3):
            SummonRepo.insert_summon(conn, dict(
                match_id="s-counts", timestamp_sec=0.0,
                wave_number=1, unit_summoned="archer", trigger_type="manual",
            ))
        for _ in range(7):
            SummonRepo.insert_summon(conn, dict(
                match_id="s-counts", timestamp_sec=0.0,
                wave_number=1, unit_summoned="knight", trigger_type="manual",
            ))
        rows = SummonRepo.get_unit_summon_counts(conn)
        counts = {r["unit_summoned"]: r["count"] for r in rows}
        assert counts["archer"] == 3
        assert counts["knight"] == 7

    def test_get_session_count_zero_on_empty_db(self, summon_analysis_conn):
        assert SummonRepo.get_session_count(summon_analysis_conn) == 0

    def test_get_session_count_counts_only_confirmed(self, summon_analysis_conn):
        conn = summon_analysis_conn
        SummonRepo.open_session(conn, "s-unconf")        # no deck_json
        self._open(conn, "s-conf1", deck=["archer"])    # confirmed
        self._open(conn, "s-conf2", deck=["knight"])    # confirmed
        assert SummonRepo.get_session_count(conn) == 2

    def test_get_total_summon_count_zero_on_empty(self, summon_analysis_conn):
        assert SummonRepo.get_total_summon_count(summon_analysis_conn) == 0

    def test_get_total_summon_count_sums_confirmed_sessions(self, summon_analysis_conn):
        conn = summon_analysis_conn
        SummonRepo.open_session(conn, "s-t1")
        SummonRepo.close_session(conn, "s-t1", json.dumps(["archer"]), 40, 5)
        SummonRepo.open_session(conn, "s-t2")
        SummonRepo.close_session(conn, "s-t2", json.dumps(["knight"]), 60, 3)
        assert SummonRepo.get_total_summon_count(conn) == 100

    def test_get_unit_merge_counts(self, summon_analysis_conn):
        conn = summon_analysis_conn
        self._open(conn, "s-mc", deck=["archer"])
        SummonRepo.insert_merge(conn, dict(
            match_id="s-mc", timestamp_sec=0.0, wave_number=1,
            unit_id="archer", from_rank=2, to_rank=3,
        ))
        SummonRepo.insert_merge(conn, dict(
            match_id="s-mc", timestamp_sec=1.0, wave_number=2,
            unit_id="archer", from_rank=2, to_rank=3,
        ))
        rows = SummonRepo.get_unit_merge_counts(conn)
        assert len(rows) == 1
        assert rows[0]["unit_id"] == "archer"
        assert rows[0]["from_rank"] == 2
        assert rows[0]["count"] == 2
