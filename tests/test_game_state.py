"""
Tests for GameState, BoardState, and UnitCell.

Covers:
  - UnitCell property logic (highest_talent_tier, branch_confidence)
  - BoardState grid access, occupied(), unit_ids()
  - GameState.to_snapshot_dict() serialization
"""

import json

import pytest

from src.analysis.game_state import BoardState, GameState, UnitCell
from tests.conftest import make_board, make_cell, make_state


# ===========================================================================
# UnitCell
# ===========================================================================

class TestUnitCell:

    def test_default_values(self):
        cell = UnitCell(unit_id="archer", merge_rank=1)
        assert cell.talent_path == {}
        assert cell.appearance_state == "base"
        assert cell.variant_tag is None
        assert cell.recognition_confidence == 0.0

    def test_highest_talent_tier_none_when_no_talents(self):
        cell = make_cell(talent_path={})
        assert cell.highest_talent_tier is None

    def test_highest_talent_tier_single(self):
        cell = make_cell(talent_path={1: "L"})
        assert cell.highest_talent_tier == 1

    def test_highest_talent_tier_multiple(self):
        cell = make_cell(talent_path={1: "R", 2: "Fixed", 3: "L"})
        assert cell.highest_talent_tier == 3

    def test_branch_confidence_no_talents(self):
        # No talent path → all "resolved" by convention (nothing to resolve)
        cell = make_cell(talent_path={})
        assert cell.branch_confidence == 1.0

    def test_branch_confidence_fully_resolved(self):
        cell = make_cell(talent_path={1: "L", 2: "R"})
        assert cell.branch_confidence == 1.0

    def test_branch_confidence_partially_resolved(self):
        # 2 of 3 tiers have a known branch
        cell = make_cell(talent_path={1: None, 2: "R", 3: "L"})
        assert cell.branch_confidence == pytest.approx(2 / 3)

    def test_branch_confidence_none_resolved(self):
        cell = make_cell(talent_path={1: None, 2: None})
        assert cell.branch_confidence == 0.0

    def test_merge_rank_stored_correctly(self):
        cell = make_cell(merge_rank=5)
        assert cell.merge_rank == 5

    def test_variant_tag_stored(self):
        cell = UnitCell(unit_id="twins", merge_rank=2, variant_tag="moon")
        assert cell.variant_tag == "moon"


# ===========================================================================
# BoardState
# ===========================================================================

class TestBoardState:

    def test_empty_board_has_no_occupied_cells(self):
        board = BoardState()
        assert board.occupied() == []

    def test_get_returns_none_on_empty_cell(self):
        board = BoardState()
        assert board.get(0, 0) is None

    def test_set_and_get_round_trip(self):
        board = BoardState()
        cell = make_cell("archer", 2)
        board.set(2, 1, cell)
        assert board.get(2, 1) is cell

    def test_set_none_clears_cell(self):
        board = BoardState()
        board.set(0, 0, make_cell())
        board.set(0, 0, None)
        assert board.get(0, 0) is None

    def test_occupied_returns_all_filled_cells(self):
        board = make_board(
            (0, 0, make_cell("a")),
            (3, 2, make_cell("b")),
        )
        occupied = board.occupied()
        assert len(occupied) == 2
        coords = {(r, c) for r, c, _ in occupied}
        assert coords == {(0, 0), (3, 2)}

    def test_occupied_returns_correct_cells(self):
        cell = make_cell("knight", 3)
        board = make_board((1, 1, cell))
        occ = board.occupied()
        assert len(occ) == 1
        r, c, returned_cell = occ[0]
        assert r == 1
        assert c == 1
        assert returned_cell is cell

    def test_unit_ids_empty_board(self):
        assert BoardState().unit_ids() == set()

    def test_unit_ids_single_unit(self):
        board = make_board((0, 0, make_cell("chemist")))
        assert board.unit_ids() == {"chemist"}

    def test_unit_ids_multiple_unique(self):
        board = make_board(
            (0, 0, make_cell("archer")),
            (1, 0, make_cell("knight")),
            (2, 0, make_cell("chemist")),
        )
        assert board.unit_ids() == {"archer", "knight", "chemist"}

    def test_unit_ids_deduplicates_same_unit(self):
        # Two archer cells → still one unique id
        board = make_board(
            (0, 0, make_cell("archer", merge_rank=1)),
            (0, 1, make_cell("archer", merge_rank=2)),
        )
        assert board.unit_ids() == {"archer"}

    def test_board_is_3x5(self):
        board = BoardState()
        assert len(board.cells) == 5
        for row in board.cells:
            assert len(row) == 3

    def test_all_corners_accessible(self):
        board = BoardState()
        for row in (0, 4):
            for col in (0, 2):
                board.set(row, col, make_cell())
        assert len(board.occupied()) == 4


# ===========================================================================
# GameState
# ===========================================================================

class TestGameState:

    def test_default_boards_are_empty(self):
        state = GameState()
        assert state.player_board.occupied() == []
        assert state.opponent_board.occupied() == []

    def test_hp_values_stored(self):
        state = make_state(player_hp=3, opponent_hp=1)
        assert state.player_hp == 3
        assert state.opponent_hp == 1

    def test_wave_number_stored(self):
        state = make_state(wave_number=12)
        assert state.wave_number == 12

    def test_win_probability_starts_none(self):
        state = GameState()
        assert state.win_probability is None

    # --- to_snapshot_dict ---

    def test_snapshot_dict_has_required_keys(self):
        state = make_state()
        snap = state.to_snapshot_dict()
        required = {
            "match_id", "timestamp_sec", "wave_number",
            "player_hp", "opponent_hp", "player_mana",
            "player_board", "opponent_board",
            "active_buffs", "win_probability", "confidence",
        }
        assert required.issubset(snap.keys())

    def test_snapshot_dict_empty_boards_serialize_as_empty_json_arrays(self):
        state = make_state()
        snap = state.to_snapshot_dict()
        assert json.loads(snap["player_board"]) == []
        assert json.loads(snap["opponent_board"]) == []

    def test_snapshot_dict_board_serializes_cells(self):
        board = make_board(
            (0, 0, make_cell("archer", merge_rank=3, confidence=0.85)),
        )
        state = make_state(player_board=board)
        snap = state.to_snapshot_dict()
        cells = json.loads(snap["player_board"])
        assert len(cells) == 1
        c = cells[0]
        assert c["unit_id"] == "archer"
        assert c["rank"] == 3
        assert c["confidence"] == pytest.approx(0.85, abs=0.001)
        assert c["cell"] == "0,0"

    def test_snapshot_dict_talent_path_included(self):
        cell = make_cell("knight", talent_path={1: "L", 2: "R"})
        board = make_board((0, 0, cell))
        state = make_state(player_board=board)
        snap = state.to_snapshot_dict()
        cells = json.loads(snap["player_board"])
        assert cells[0]["talent_tier"] == 2
        # talent_path is stored as a dict (JSON object) — keys are ints (or strings after JSON round-trip)
        stored_path = cells[0]["talent_path"]
        assert stored_path is not None

    def test_snapshot_dict_active_buffs_serializes(self):
        state = make_state(active_buffs={"archer": ["fire_aura"]})
        snap = state.to_snapshot_dict()
        buffs = json.loads(snap["active_buffs"])
        assert buffs == {"archer": ["fire_aura"]}

    def test_snapshot_dict_match_id_passed_through(self):
        state = make_state(match_id="match-xyz-999")
        snap = state.to_snapshot_dict()
        assert snap["match_id"] == "match-xyz-999"

    def test_snapshot_dict_confidence_rounded(self):
        state = make_state()
        state.pipeline_confidence = 0.876543
        snap = state.to_snapshot_dict()
        # Should be rounded to 3 decimal places
        assert snap["confidence"] == pytest.approx(0.877, abs=0.0005)

    def test_snapshot_dict_win_probability_none_when_unset(self):
        state = make_state()
        snap = state.to_snapshot_dict()
        assert snap["win_probability"] is None

    def test_snapshot_dict_win_probability_set(self):
        state = make_state()
        state.win_probability = 0.72
        snap = state.to_snapshot_dict()
        assert snap["win_probability"] == pytest.approx(0.72)
