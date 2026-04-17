"""
Tests for SummonAnalyzer — the statistical engine that tests whether
Rush Royale's summon distribution is truly uniform (20% per unit).

Covers:
  - No-data path (empty DB)
  - Per-unit stats computation (observed rate, Z-score, CI)
  - Chi-squared statistic construction
  - Verdict strings for reliable vs. insufficient data
  - Merge stat aggregation
  - trigger_type filtering (all / manual / post_merge)
  - Statistical helpers: Wilson CI bounds, normal p-value symmetry

All tests use the in-memory summon_analysis.db fixture from conftest.
"""

import json

import pytest

from src.analysis.summon_analyzer import (
    SummonAnalyzer,
    _wilson_ci,
    _normal_p_value,
)


# ---------------------------------------------------------------------------
# Helpers to seed the in-memory DB
# ---------------------------------------------------------------------------

DECK = ["archer", "knight", "chemist", "monk", "rogue"]


def _open_session(conn, match_id: str, deck=None):
    from datetime import datetime, timezone
    conn.execute(
        """INSERT OR IGNORE INTO summon_sessions
           (match_id, recorded_at, total_summons, total_merges)
           VALUES (?, ?, 0, 0)""",
        (match_id, datetime.now(timezone.utc).isoformat()),
    )
    if deck is not None:
        conn.execute(
            "UPDATE summon_sessions SET deck_json=? WHERE match_id=?",
            (json.dumps(sorted(deck)), match_id),
        )


def _insert_summons(conn, match_id: str, counts: dict[str, int],
                    trigger_type: str = "manual"):
    """Insert `counts` summon events for each unit_id."""
    ts = 0.0
    for unit_id, n in counts.items():
        for _ in range(n):
            conn.execute(
                """INSERT INTO summon_events
                   (match_id, timestamp_sec, wave_number, unit_summoned, trigger_type)
                   VALUES (?, ?, 1, ?, ?)""",
                (match_id, ts, unit_id, trigger_type),
            )
            ts += 1.0
    conn.commit()


def _insert_merges(conn, match_id: str, unit_id: str,
                   from_rank: int, to_rank: int, count: int = 1):
    for i in range(count):
        conn.execute(
            """INSERT INTO merge_events
               (match_id, timestamp_sec, wave_number, unit_id, from_rank, to_rank)
               VALUES (?, ?, 1, ?, ?, ?)""",
            (match_id, float(i), unit_id, from_rank, to_rank),
        )
    conn.commit()


def _seed_uniform(conn, n_per_unit: int = 20,
                  trigger_type: str = "manual") -> str:
    """
    Seed a single session with exactly n_per_unit summons for each of the
    5 deck units (perfectly uniform distribution).
    Returns the match_id used.
    """
    match_id = "match-uniform-001"
    _open_session(conn, match_id, deck=DECK)
    counts = {u: n_per_unit for u in DECK}
    _insert_summons(conn, match_id, counts, trigger_type=trigger_type)
    return match_id


# ===========================================================================
# No-data / empty DB
# ===========================================================================

class TestNoData:

    def test_empty_db_returns_no_data_verdict(self, summon_analysis_conn):
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.verdict == "No data yet"
        assert result.total_summons == 0

    def test_empty_db_unit_stats_empty(self, summon_analysis_conn):
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.unit_stats == []

    def test_session_without_deck_json_excluded(self, summon_analysis_conn):
        # Open a session but don't confirm deck — events must be excluded
        conn = summon_analysis_conn
        _open_session(conn, "no-deck-match", deck=None)
        _insert_summons(conn, "no-deck-match", {"archer": 10})
        result = SummonAnalyzer.analyse(conn)
        assert result.total_summons == 0
        assert result.verdict == "No data yet"


# ===========================================================================
# Unit stats computation
# ===========================================================================

class TestUnitStats:

    def test_unit_count_matches_deck_size(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert len(result.unit_stats) == len(DECK)

    def test_unit_ids_cover_full_deck(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        ids = {u.unit_id for u in result.unit_stats}
        assert ids == set(DECK)

    def test_uniform_distribution_observed_rate_near_one_fifth(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert u.observed_rate == pytest.approx(0.2, abs=0.001)

    def test_expected_rate_is_one_over_deck_size(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert u.expected_rate == pytest.approx(1.0 / 5, rel=1e-4)

    def test_z_score_near_zero_for_uniform(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert abs(u.z_score) < 0.1

    def test_flagged_false_for_uniform_distribution(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=40)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert not u.flagged

    def test_biased_unit_gets_higher_observed_rate(self, summon_analysis_conn):
        match_id = "biased-match"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        # archer gets 4× more summons than others
        counts = {"archer": 80, "knight": 20, "chemist": 20, "monk": 20, "rogue": 20}
        _insert_summons(summon_analysis_conn, match_id, counts)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        archer_stats = next(u for u in result.unit_stats if u.unit_id == "archer")
        others = [u for u in result.unit_stats if u.unit_id != "archer"]
        assert archer_stats.observed_rate > max(u.observed_rate for u in others)

    def test_biased_unit_flagged(self, summon_analysis_conn):
        match_id = "biased-flagged"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        # Extremely biased: one unit gets nearly all summons
        counts = {"archer": 190, "knight": 1, "chemist": 1, "monk": 1, "rogue": 7}
        _insert_summons(summon_analysis_conn, match_id, counts)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        archer_stats = next(u for u in result.unit_stats if u.unit_id == "archer")
        assert archer_stats.flagged

    def test_wilson_ci_contains_true_rate_for_uniform(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=40)  # n=200 total
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert u.ci_low <= 0.2 <= u.ci_high

    def test_ci_bounds_ordered(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            assert u.ci_low <= u.observed_rate <= u.ci_high

    def test_total_summons_correct(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)  # 5*20 = 100
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.total_summons == 100

    def test_deviation_pct_property(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        for u in result.unit_stats:
            expected_dev = (u.observed_rate - u.expected_rate) * 100.0
            assert u.deviation_pct == pytest.approx(expected_dev, abs=0.001)


# ===========================================================================
# Chi-squared and verdict
# ===========================================================================

class TestChiSquaredAndVerdict:

    def test_chi_sq_near_zero_for_uniform(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.chi_sq_statistic < 1.0

    def test_chi_sq_df_is_deck_size_minus_one(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.chi_sq_df == len(DECK) - 1

    def test_chi_sq_p_value_high_for_uniform(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=40)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.chi_sq_p_value > 0.05

    def test_insufficient_data_verdict(self, summon_analysis_conn):
        # Seed only a small amount — below MIN_RELIABLE (100)
        match_id = "small-match"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id, {u: 5 for u in DECK})  # 25 total
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert not result.reliable
        assert "data" in result.verdict.lower() or "collecting" in result.verdict.lower()

    def test_reliable_flag_true_at_100_summons(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=20)  # exactly 100
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.reliable

    def test_uniform_verdict_is_consistent_with_fair(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn, n_per_unit=40)  # 200 summons, perfectly even
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.reliable
        assert "fair" in result.verdict.lower() or "consistent" in result.verdict.lower()

    def test_heavily_biased_verdict_signals_bias(self, summon_analysis_conn):
        match_id = "heavy-bias"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        # n=500 total, one unit gets 80% of summons → extremely significant
        counts = {"archer": 400, "knight": 25, "chemist": 25, "monk": 25, "rogue": 25}
        _insert_summons(summon_analysis_conn, match_id, counts)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.reliable
        assert result.chi_sq_p_value < 0.01
        assert "bias" in result.verdict.lower() or "suspicious" in result.verdict.lower()


# ===========================================================================
# Trigger-type filtering
# ===========================================================================

class TestTriggerTypeFilter:

    def test_manual_filter_excludes_post_merge_events(self, summon_analysis_conn):
        match_id = "trigger-test"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id,
                        {"archer": 10}, trigger_type="manual")
        _insert_summons(summon_analysis_conn, match_id,
                        {"knight": 10}, trigger_type="post_merge")

        result = SummonAnalyzer.analyse(summon_analysis_conn, trigger_type="manual")
        # Only manual events → only archer should appear in counts
        unit_counts = {u.unit_id: u.observed for u in result.unit_stats}
        assert unit_counts.get("archer", 0) == 10
        assert unit_counts.get("knight", 0) == 0

    def test_post_merge_filter_excludes_manual_events(self, summon_analysis_conn):
        match_id = "trigger-test-2"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id,
                        {"archer": 5}, trigger_type="manual")
        _insert_summons(summon_analysis_conn, match_id,
                        {"rogue": 8}, trigger_type="post_merge")

        result = SummonAnalyzer.analyse(summon_analysis_conn, trigger_type="post_merge")
        unit_counts = {u.unit_id: u.observed for u in result.unit_stats}
        assert unit_counts.get("rogue", 0) == 8
        assert unit_counts.get("archer", 0) == 0

    def test_all_trigger_includes_both_types(self, summon_analysis_conn):
        match_id = "trigger-all"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id,
                        {"archer": 5}, trigger_type="manual")
        _insert_summons(summon_analysis_conn, match_id,
                        {"archer": 3}, trigger_type="post_merge")

        result = SummonAnalyzer.analyse(summon_analysis_conn, trigger_type="all")
        unit_counts = {u.unit_id: u.observed for u in result.unit_stats}
        assert unit_counts.get("archer", 0) == 8


# ===========================================================================
# Merge stats
# ===========================================================================

class TestMergeStats:

    def test_merge_stats_empty_when_no_merges(self, summon_analysis_conn):
        _seed_uniform(summon_analysis_conn)
        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert result.merge_stats == []

    def test_merge_stats_populated_correctly(self, summon_analysis_conn):
        match_id = "merge-test"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id, {"archer": 20})
        _insert_merges(summon_analysis_conn, match_id, "archer",
                       from_rank=1, to_rank=2, count=3)

        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert len(result.merge_stats) == 1
        ms = result.merge_stats[0]
        assert ms.unit_id == "archer"
        assert ms.from_rank == 1
        assert ms.to_rank == 2
        assert ms.count == 3

    def test_multiple_merge_types_all_returned(self, summon_analysis_conn):
        match_id = "multi-merge"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id, {"archer": 20, "knight": 20})
        _insert_merges(summon_analysis_conn, match_id, "archer", 1, 2, count=4)
        _insert_merges(summon_analysis_conn, match_id, "archer", 2, 3, count=2)
        _insert_merges(summon_analysis_conn, match_id, "knight", 1, 2, count=5)

        result = SummonAnalyzer.analyse(summon_analysis_conn)
        assert len(result.merge_stats) == 3

    def test_unit_name_filled_from_unit_names_map(self, summon_analysis_conn):
        match_id = "names-test"
        _open_session(summon_analysis_conn, match_id, deck=DECK)
        _insert_summons(summon_analysis_conn, match_id, {"archer": 20})
        _insert_merges(summon_analysis_conn, match_id, "archer", 1, 2, count=1)

        result = SummonAnalyzer.analyse(
            summon_analysis_conn,
            unit_names={"archer": "Archer of Light"},
        )
        ms = result.merge_stats[0]
        assert ms.display_name == "Archer of Light"


# ===========================================================================
# Statistical helper unit tests
# ===========================================================================

class TestWilsonCI:

    def test_zero_observations_gives_low_bound_near_zero(self):
        # With k=0, n=10 at 95% CI the Wilson upper bound is ~0.28 —
        # it is NOT 1.0 because Wilson applies a correction that pulls
        # the centre away from 0 and caps uncertainty at the 95% level.
        lo, hi = _wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=0.01)
        assert 0.0 < hi < 0.5

    def test_all_observations_returns_near_one(self):
        lo, hi = _wilson_ci(100, 100)
        assert lo > 0.9
        assert hi == pytest.approx(1.0, abs=0.01)

    def test_zero_total_returns_full_range(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_bounds_within_unit_interval(self):
        for k, n in [(0, 10), (5, 10), (10, 10), (1, 100), (50, 100)]:
            lo, hi = _wilson_ci(k, n)
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0

    def test_lower_bound_less_than_upper(self):
        for k, n in [(1, 10), (5, 10), (9, 10)]:
            lo, hi = _wilson_ci(k, n)
            assert lo < hi

    def test_symmetric_around_half_at_five_of_ten(self):
        lo, hi = _wilson_ci(5, 10)
        assert lo == pytest.approx(1.0 - hi, abs=0.001)


class TestNormalPValue:

    def test_z_zero_gives_p_one(self):
        assert _normal_p_value(0.0) == pytest.approx(1.0, abs=0.001)

    def test_z_large_positive_gives_small_p(self):
        assert _normal_p_value(5.0) < 0.0001

    def test_z_large_negative_gives_small_p(self):
        assert _normal_p_value(-5.0) < 0.0001

    def test_symmetry(self):
        assert _normal_p_value(2.0) == pytest.approx(_normal_p_value(-2.0), rel=1e-4)

    def test_known_value_z_196(self):
        # z=1.96 → p ≈ 0.05 (two-tailed)
        assert _normal_p_value(1.96) == pytest.approx(0.05, abs=0.005)

    def test_p_value_in_unit_interval(self):
        for z in (-3.0, -1.96, -1.0, 0.0, 1.0, 1.96, 3.0):
            p = _normal_p_value(z)
            assert 0.0 <= p <= 1.0
