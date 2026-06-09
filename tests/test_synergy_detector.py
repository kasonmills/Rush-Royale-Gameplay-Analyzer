"""
Tests for SynergyDetector and the synergy component of WinPredictor.

Covers:
  - _parse_strength(): numeric, keyword, mixed, unknown
  - SynergyDetector.detect(): absent units, one unit only, non-positional
    presence, positional adjacent, positional non-adjacent, diagonal excluded
  - SynergyDetector.load_from_csv(): loads known pairs, positional flag
  - SynergyDetector.load_from_db(): loads from in-memory DB
  - SynergyDetector.known_pairs() and entry_count()
  - WinPredictor._compute_synergy_advantage(): no detector, player/opponent
    advantage, equal synergies, partial overlap
  - WinPredictor.predict() still returns [0, 1] with synergy detector wired
"""

from pathlib import Path

import pytest

from src.analysis.game_state import BoardState
from src.analysis.synergy_detector import (
    SynergyDetector,
    SynergyEntry,
    SynergyResult,
    _parse_strength,
)
from src.analysis.win_predictor import WinPredictor
from tests.conftest import make_board, make_cell, make_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNERGY_CSV = PROJECT_ROOT / "data" / "Synergies.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(*entries: SynergyEntry) -> SynergyDetector:
    det = SynergyDetector()
    det._entries = list(entries)
    return det


def _nonpositional(unit_a: str, unit_b: str,
                   strength: float = 0.5) -> SynergyEntry:
    return SynergyEntry(unit_a, unit_b, f"{unit_a}+{unit_b}", "DPS", "",
                        strength, False, "Done")


def _positional(unit_a: str, unit_b: str,
                strength: float = 1.0) -> SynergyEntry:
    return SynergyEntry(unit_a, unit_b, f"{unit_a}+{unit_b}", "DPS", "",
                        strength, True, "Done")


# ---------------------------------------------------------------------------
# _parse_strength
# ---------------------------------------------------------------------------

class TestParseStrength:

    def test_numeric_string(self):
        assert _parse_strength("0.8") == pytest.approx(0.8)

    def test_one_point_zero(self):
        assert _parse_strength("1.0") == pytest.approx(1.0)

    def test_high_keyword(self):
        assert _parse_strength("High") == pytest.approx(1.0)

    def test_high_with_qualifier(self):
        assert _parse_strength("High (TBD exact %)") == pytest.approx(1.0)

    def test_med_keyword(self):
        assert _parse_strength("Med") == pytest.approx(0.5)

    def test_medium_keyword(self):
        assert _parse_strength("Medium") == pytest.approx(0.5)

    def test_low_keyword(self):
        assert _parse_strength("Low") == pytest.approx(0.25)

    def test_unknown_defaults_to_medium(self):
        assert _parse_strength("TBD") == pytest.approx(0.5)

    def test_empty_defaults_to_medium(self):
        assert _parse_strength("") == pytest.approx(0.5)

    def test_clamped_above_one(self):
        assert _parse_strength("2.5") == pytest.approx(1.0)

    def test_clamped_below_zero(self):
        assert _parse_strength("-0.5") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SynergyDetector.detect — absence / presence
# ---------------------------------------------------------------------------

class TestDetectAbsenceAndPresence:

    def test_empty_board_returns_empty(self):
        det = _make_detector(_nonpositional("inquisitor", "knight_statue"))
        assert det.detect(BoardState()) == []

    def test_only_unit_a_present_returns_empty(self):
        det = _make_detector(_nonpositional("inquisitor", "knight_statue"))
        board = make_board((0, 0, make_cell("inquisitor")))
        assert det.detect(board) == []

    def test_only_unit_b_present_returns_empty(self):
        det = _make_detector(_nonpositional("inquisitor", "knight_statue"))
        board = make_board((0, 0, make_cell("knight_statue")))
        assert det.detect(board) == []

    def test_both_present_nonpositional_returns_result(self):
        det = _make_detector(_nonpositional("engineer", "harlequin", strength=0.8))
        board = make_board(
            (0, 0, make_cell("engineer")),
            (2, 4, make_cell("harlequin")),  # far apart — shouldn't matter
        )
        results = det.detect(board)
        assert len(results) == 1
        assert results[0].unit_a_id == "engineer"
        assert results[0].unit_b_id == "harlequin"
        assert results[0].strength_bonus == pytest.approx(0.8)

    def test_unrelated_units_on_board_no_false_positive(self):
        det = _make_detector(_nonpositional("inquisitor", "knight_statue"))
        board = make_board(
            (0, 0, make_cell("archer")),
            (0, 1, make_cell("rogue")),
        )
        assert det.detect(board) == []

    def test_multiple_synergies_all_detected(self):
        det = _make_detector(
            _nonpositional("engineer", "harlequin"),
            _nonpositional("archer", "rogue"),
        )
        board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
            (1, 0, make_cell("archer")),
            (1, 1, make_cell("rogue")),
        )
        results = det.detect(board)
        assert len(results) == 2

    def test_only_matching_synergy_detected_when_both_present(self):
        det = _make_detector(
            _nonpositional("engineer", "harlequin"),
            _nonpositional("archer", "rogue"),
        )
        board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
            # archer without rogue
        )
        results = det.detect(board)
        assert len(results) == 1
        assert results[0].unit_a_id == "engineer"


# ---------------------------------------------------------------------------
# SynergyDetector.detect — positional (adjacency)
# ---------------------------------------------------------------------------

class TestDetectPositional:

    def test_positional_adjacent_horizontal_detected(self):
        det = _make_detector(_positional("inquisitor", "knight_statue"))
        board = make_board(
            (2, 0, make_cell("inquisitor")),
            (2, 1, make_cell("knight_statue")),  # same row, next column
        )
        assert len(det.detect(board)) == 1

    def test_positional_adjacent_vertical_detected(self):
        det = _make_detector(_positional("inquisitor", "knight_statue"))
        board = make_board(
            (1, 1, make_cell("inquisitor")),
            (2, 1, make_cell("knight_statue")),  # same column, next row
        )
        assert len(det.detect(board)) == 1

    def test_positional_diagonal_not_adjacent(self):
        det = _make_detector(_positional("inquisitor", "knight_statue"))
        board = make_board(
            (1, 0, make_cell("inquisitor")),
            (2, 1, make_cell("knight_statue")),  # diagonal — Manhattan = 2
        )
        assert det.detect(board) == []

    def test_positional_two_apart_not_adjacent(self):
        det = _make_detector(_positional("inquisitor", "knight_statue"))
        board = make_board(
            (0, 0, make_cell("inquisitor")),
            (2, 0, make_cell("knight_statue")),  # same column, 2 rows apart
        )
        assert det.detect(board) == []

    def test_positional_both_present_far_apart_no_result(self):
        det = _make_detector(_positional("inquisitor", "knight_statue"))
        board = make_board(
            (0, 0, make_cell("inquisitor")),
            (2, 4, make_cell("knight_statue")),  # opposite corners
        )
        assert det.detect(board) == []

    def test_nonpositional_far_apart_still_detected(self):
        det = _make_detector(_nonpositional("inquisitor", "knight_statue"))
        board = make_board(
            (0, 0, make_cell("inquisitor")),
            (2, 4, make_cell("knight_statue")),
        )
        assert len(det.detect(board)) == 1

    def test_positional_multiple_copies_one_adjacent_pair_sufficient(self):
        """If a unit appears multiple times, only one adjacent pair needed."""
        det = _make_detector(_positional("engineer", "harlequin"))
        board = make_board(
            (0, 0, make_cell("engineer")),   # not adjacent to harlequin
            (2, 0, make_cell("engineer")),   # adjacent to harlequin at (2,1)
            (2, 1, make_cell("harlequin")),
        )
        assert len(det.detect(board)) == 1


# ---------------------------------------------------------------------------
# SynergyDetector.load_from_csv
# ---------------------------------------------------------------------------

class TestLoadFromCSV:

    def _load(self) -> SynergyDetector:
        if not SYNERGY_CSV.exists():
            pytest.skip("data/Synergies.csv not found")
        det = SynergyDetector()
        det.load_from_csv(SYNERGY_CSV)
        return det

    def test_known_pairs_nonempty(self):
        det = self._load()
        assert det.entry_count() > 0

    def test_inquisitor_knight_statue_loaded(self):
        det = self._load()
        pair = frozenset({"inquisitor", "knight_statue"})
        assert pair in det.known_pairs()

    def test_engineer_harlequin_loaded(self):
        det = self._load()
        pair = frozenset({"engineer", "harlequin"})
        assert pair in det.known_pairs()

    def test_inquisitor_knight_statue_is_positional(self):
        det = self._load()
        for entry in det._entries:
            if {entry.unit_a_id, entry.unit_b_id} == {"inquisitor", "knight_statue"}:
                assert entry.positional is True
                return
        pytest.fail("inquisitor+knight_statue entry not found")

    def test_engineer_harlequin_is_not_positional(self):
        det = self._load()
        for entry in det._entries:
            if {entry.unit_a_id, entry.unit_b_id} == {"engineer", "harlequin"}:
                assert entry.positional is False
                return
        pytest.fail("engineer+harlequin entry not found")

    def test_strength_bonus_is_float_in_range(self):
        det = self._load()
        for entry in det._entries:
            assert 0.0 <= entry.strength_bonus <= 1.0

    def test_empty_rows_skipped(self):
        det = self._load()
        for entry in det._entries:
            assert entry.unit_a_id != ""
            assert entry.unit_b_id != ""


# ---------------------------------------------------------------------------
# SynergyDetector.load_from_db
# ---------------------------------------------------------------------------

class TestLoadFromDB:

    def _make_db(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE units (unit_id TEXT PRIMARY KEY, display_name TEXT NOT NULL);
            CREATE TABLE synergies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit_a_id TEXT NOT NULL,
                unit_b_id TEXT NOT NULL,
                description TEXT,
                strength_bonus REAL,
                positional INTEGER NOT NULL DEFAULT 0,
                research_status TEXT NOT NULL DEFAULT 'Not Started'
            );
        """)
        conn.execute("INSERT INTO units VALUES ('inquisitor', 'Inquisitor')")
        conn.execute("INSERT INTO units VALUES ('knight_statue', 'Knight Statue')")
        conn.execute(
            "INSERT INTO synergies (unit_a_id, unit_b_id, description, "
            "strength_bonus, positional) VALUES (?, ?, ?, ?, ?)",
            ("inquisitor", "knight_statue", "Adjacency buff", 1.0, 1)
        )
        conn.commit()
        return conn

    def test_entry_loaded(self):
        conn = self._make_db()
        det = SynergyDetector()
        det.load_from_db(conn)
        assert det.entry_count() == 1

    def test_pair_in_known_pairs(self):
        conn = self._make_db()
        det = SynergyDetector()
        det.load_from_db(conn)
        assert frozenset({"inquisitor", "knight_statue"}) in det.known_pairs()

    def test_positional_flag_preserved(self):
        conn = self._make_db()
        det = SynergyDetector()
        det.load_from_db(conn)
        assert det._entries[0].positional is True

    def test_strength_bonus_preserved(self):
        conn = self._make_db()
        det = SynergyDetector()
        det.load_from_db(conn)
        assert det._entries[0].strength_bonus == pytest.approx(1.0)

    def test_null_strength_defaults_to_half(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE units (unit_id TEXT PRIMARY KEY, display_name TEXT NOT NULL);
            CREATE TABLE synergies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit_a_id TEXT NOT NULL,
                unit_b_id TEXT NOT NULL,
                description TEXT,
                strength_bonus REAL,
                positional INTEGER NOT NULL DEFAULT 0,
                research_status TEXT NOT NULL DEFAULT 'Not Started'
            );
        """)
        conn.execute("INSERT INTO units VALUES ('a', 'A')")
        conn.execute("INSERT INTO units VALUES ('b', 'B')")
        conn.execute(
            "INSERT INTO synergies (unit_a_id, unit_b_id, strength_bonus) "
            "VALUES ('a', 'b', NULL)"
        )
        conn.commit()
        det = SynergyDetector()
        det.load_from_db(conn)
        assert det._entries[0].strength_bonus == pytest.approx(0.5)

    def test_load_clears_previous_entries(self):
        conn = self._make_db()
        det = SynergyDetector()
        det._entries = [_nonpositional("stale_a", "stale_b")]
        det.load_from_db(conn)
        assert all(e.unit_a_id != "stale_a" for e in det._entries)


# ---------------------------------------------------------------------------
# SynergyDetector.known_pairs / entry_count
# ---------------------------------------------------------------------------

class TestKnownPairs:

    def test_empty_registry_empty_set(self):
        assert SynergyDetector().known_pairs() == set()

    def test_single_entry_one_pair(self):
        det = _make_detector(_nonpositional("a", "b"))
        assert frozenset({"a", "b"}) in det.known_pairs()

    def test_pair_is_symmetric(self):
        det = _make_detector(_nonpositional("a", "b"))
        pairs = det.known_pairs()
        assert frozenset({"b", "a"}) in pairs   # frozenset is order-independent

    def test_entry_count_matches(self):
        det = _make_detector(
            _nonpositional("a", "b"),
            _nonpositional("c", "d"),
        )
        assert det.entry_count() == 2


# ---------------------------------------------------------------------------
# WinPredictor._compute_synergy_advantage
# ---------------------------------------------------------------------------

class TestComputeSynergyAdvantage:

    def test_no_detector_returns_zero(self):
        predictor = WinPredictor()
        state = make_state()
        assert predictor._compute_synergy_advantage(state) == pytest.approx(0.0)

    def test_no_active_synergies_returns_zero(self):
        det = _make_detector(_nonpositional("engineer", "harlequin"))
        predictor = WinPredictor(synergy_detector=det)
        state = make_state()  # empty boards
        assert predictor._compute_synergy_advantage(state) == pytest.approx(0.0)

    def test_player_synergy_gives_positive_advantage(self):
        det = _make_detector(_nonpositional("engineer", "harlequin", strength=1.0))
        predictor = WinPredictor(synergy_detector=det)
        p_board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        state = make_state(player_board=p_board)
        adv = predictor._compute_synergy_advantage(state)
        assert adv == pytest.approx(1.0)   # player 1.0, opponent 0.0

    def test_opponent_synergy_gives_negative_advantage(self):
        det = _make_detector(_nonpositional("engineer", "harlequin", strength=1.0))
        predictor = WinPredictor(synergy_detector=det)
        o_board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        state = make_state(opponent_board=o_board)
        adv = predictor._compute_synergy_advantage(state)
        assert adv == pytest.approx(-1.0)   # player 0.0, opponent 1.0

    def test_equal_synergies_both_sides_returns_zero(self):
        det = _make_detector(_nonpositional("engineer", "harlequin", strength=1.0))
        predictor = WinPredictor(synergy_detector=det)
        board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        state = make_state(player_board=board, opponent_board=board)
        adv = predictor._compute_synergy_advantage(state)
        assert adv == pytest.approx(0.0)

    def test_advantage_normalised_between_minus_one_and_one(self):
        det = _make_detector(
            _nonpositional("engineer", "harlequin", strength=0.8),
            _nonpositional("archer", "rogue", strength=0.4),
        )
        predictor = WinPredictor(synergy_detector=det)
        p_board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        o_board = make_board(
            (0, 0, make_cell("archer")),
            (0, 1, make_cell("rogue")),
        )
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = predictor._compute_synergy_advantage(state)
        assert -1.0 <= adv <= 1.0

    def test_positional_synergy_not_adjacent_excluded(self):
        det = _make_detector(_positional("inquisitor", "knight_statue", strength=1.0))
        predictor = WinPredictor(synergy_detector=det)
        p_board = make_board(
            (0, 0, make_cell("inquisitor")),
            (2, 4, make_cell("knight_statue")),  # both present, not adjacent
        )
        state = make_state(player_board=p_board)
        adv = predictor._compute_synergy_advantage(state)
        assert adv == pytest.approx(0.0)

    def test_positional_synergy_adjacent_included(self):
        det = _make_detector(_positional("inquisitor", "knight_statue", strength=1.0))
        predictor = WinPredictor(synergy_detector=det)
        p_board = make_board(
            (2, 0, make_cell("inquisitor")),
            (2, 1, make_cell("knight_statue")),  # adjacent
        )
        state = make_state(player_board=p_board)
        adv = predictor._compute_synergy_advantage(state)
        assert adv == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# WinPredictor.predict with synergy detector
# ---------------------------------------------------------------------------

class TestPredictWithSynergies:

    def test_predict_in_unit_interval_with_synergy_detector(self):
        det = _make_detector(_nonpositional("engineer", "harlequin"))
        predictor = WinPredictor(synergy_detector=det)
        p_board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        state = make_state(player_board=p_board)
        prob = predictor.predict(state)
        assert 0.0 <= prob <= 1.0

    def test_predict_player_active_synergy_above_no_synergy(self):
        """Active player synergy should nudge probability above the baseline."""
        det = _make_detector(_nonpositional("engineer", "harlequin", strength=1.0))
        predictor_with = WinPredictor(synergy_detector=det)
        predictor_without = WinPredictor()

        p_board = make_board(
            (0, 0, make_cell("engineer")),
            (0, 1, make_cell("harlequin")),
        )
        state = make_state(player_board=p_board)

        prob_with = predictor_with.predict(state)
        prob_without = predictor_without.predict(state)
        assert prob_with > prob_without