"""
Tests for WinPredictor and its component functions.

Covers:
  - _sigmoid helper (clamping, symmetry, known values)
  - _hp_trajectory_advantage (no HP, equal HP, player/opponent ahead)
  - _rank_efficiency_advantage (empty boards, single side, both sides)
  - _hero_advantage (no DB, missing heroes, score differential)
  - _deck_tier_advantage (no DB, one side only, full differential)
  - predict() integration (no DB, with DB, win-probability output range)

All DB interactions use in-memory SQLite from conftest fixtures.
HP range is 0–3 (each player starts with 3 lives in PvP).
"""

import pytest

from src.analysis.win_predictor import WinPredictor, _sigmoid
from tests.conftest import (
    insert_tier_score,
    make_board,
    make_cell,
    make_state,
)


# ===========================================================================
# _sigmoid helper
# ===========================================================================

class TestSigmoid:

    def test_zero_input_gives_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_positive_input_above_half(self):
        assert _sigmoid(1.0) > 0.5

    def test_negative_input_below_half(self):
        assert _sigmoid(-1.0) < 0.5

    def test_symmetry(self):
        assert _sigmoid(2.0) == pytest.approx(1.0 - _sigmoid(-2.0))

    def test_large_positive_clamps_near_one(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_clamps_near_zero(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_output_always_in_unit_interval(self):
        for x in (-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0):
            result = _sigmoid(x)
            assert 0.0 <= result <= 1.0


# ===========================================================================
# HP trajectory component
# ===========================================================================

class TestHPTrajectory:

    def setup_method(self):
        self.predictor = WinPredictor()

    def test_returns_zero_when_player_hp_none(self):
        state = make_state(player_hp=None, opponent_hp=1)
        assert self.predictor._hp_trajectory_advantage(state) == 0.0

    def test_returns_zero_when_opponent_hp_none(self):
        state = make_state(player_hp=3, opponent_hp=None)
        assert self.predictor._hp_trajectory_advantage(state) == 0.0

    def test_returns_zero_when_both_hp_none(self):
        state = make_state(player_hp=None, opponent_hp=None)
        assert self.predictor._hp_trajectory_advantage(state) == 0.0

    def test_equal_hp_gives_zero(self):
        state = make_state(player_hp=3, opponent_hp=3)
        assert self.predictor._hp_trajectory_advantage(state) == pytest.approx(0.0)

    def test_opponent_lower_hp_gives_positive_advantage(self):
        # Opponent on 1 life, player on 3 → player is winning
        state = make_state(player_hp=3, opponent_hp=1)
        adv = self.predictor._hp_trajectory_advantage(state)
        assert adv > 0.0

    def test_player_lower_hp_gives_negative_advantage(self):
        # Player on 1 life, opponent on 3 → player is losing
        state = make_state(player_hp=1, opponent_hp=3)
        adv = self.predictor._hp_trajectory_advantage(state)
        assert adv < 0.0

    def test_max_advantage_when_opponent_at_zero(self):
        # Opponent just died — maximum possible positive advantage
        state = make_state(player_hp=3, opponent_hp=0)
        adv = self.predictor._hp_trajectory_advantage(state)
        assert adv == pytest.approx(1.0)

    def test_min_advantage_when_player_at_zero(self):
        state = make_state(player_hp=0, opponent_hp=3)
        adv = self.predictor._hp_trajectory_advantage(state)
        assert adv == pytest.approx(-1.0)

    def test_one_life_difference(self):
        # opponent_hp=2, player_hp=3 → opponent lost one life, player none
        # (player_hp - opponent_hp) / 3 = (3-2)/3 = +1/3 (player slightly ahead)
        state = make_state(player_hp=3, opponent_hp=2)
        adv = self.predictor._hp_trajectory_advantage(state)
        assert adv == pytest.approx(1 / 3, rel=1e-4)


# ===========================================================================
# Rank efficiency component
# ===========================================================================

class TestRankEfficiency:

    def setup_method(self):
        self.predictor = WinPredictor()

    def test_both_empty_returns_zero(self):
        state = make_state()  # both boards empty
        assert self.predictor._rank_efficiency_advantage(state) == 0.0

    def test_player_units_only_gives_positive(self):
        board = make_board((0, 0, make_cell("archer", merge_rank=3)))
        state = make_state(player_board=board)
        assert self.predictor._rank_efficiency_advantage(state) > 0.0

    def test_opponent_units_only_gives_negative(self):
        board = make_board((0, 0, make_cell("rogue", merge_rank=3)))
        state = make_state(opponent_board=board)
        assert self.predictor._rank_efficiency_advantage(state) < 0.0

    def test_equal_total_rank_gives_zero(self):
        p_board = make_board((0, 0, make_cell("archer", merge_rank=3)))
        o_board = make_board((0, 0, make_cell("rogue",  merge_rank=3)))
        state = make_state(player_board=p_board, opponent_board=o_board)
        assert self.predictor._rank_efficiency_advantage(state) == pytest.approx(0.0)

    def test_advantage_scales_with_rank_difference(self):
        # Player has rank-7, opponent rank-1 → large positive advantage
        p_board = make_board((0, 0, make_cell("archer", merge_rank=7)))
        o_board = make_board((0, 0, make_cell("rogue",  merge_rank=1)))
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._rank_efficiency_advantage(state)
        assert adv > 0.0

    def test_advantage_capped_at_max_normalised(self):
        # Max possible: player has 5 units at rank 7 = 35 total, opponent 0
        # Expected advantage = 35/35 = 1.0
        p_board = make_board(
            (0, 0, make_cell("a", merge_rank=7)),
            (0, 1, make_cell("b", merge_rank=7)),
            (0, 2, make_cell("c", merge_rank=7)),
            (1, 0, make_cell("d", merge_rank=7)),
            (1, 1, make_cell("e", merge_rank=7)),
        )
        state = make_state(player_board=p_board)
        adv = self.predictor._rank_efficiency_advantage(state)
        assert adv == pytest.approx(1.0)

    def test_multiple_units_summed(self):
        # Player: rank 2 + rank 3 = 5; Opponent: rank 4 → diff = 1
        p_board = make_board(
            (0, 0, make_cell("a", merge_rank=2)),
            (0, 1, make_cell("b", merge_rank=3)),
        )
        o_board = make_board((0, 0, make_cell("c", merge_rank=4)))
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._rank_efficiency_advantage(state)
        # (5 - 4) / 35
        assert adv == pytest.approx(1 / 35, rel=1e-4)


# ===========================================================================
# Hero advantage component
# ===========================================================================

class TestHeroAdvantage:

    def setup_method(self):
        self.predictor = WinPredictor()

    def test_returns_zero_without_db(self):
        state = make_state(player_hero_id="paladin", opponent_hero_id="shaman")
        assert self.predictor._hero_advantage(state, None) == 0.0

    def test_returns_zero_when_neither_hero_in_db(self, unit_meta_conn):
        state = make_state(player_hero_id="unknown_a", opponent_hero_id="unknown_b")
        assert self.predictor._hero_advantage(state, unit_meta_conn) == 0.0

    def test_returns_zero_when_hero_ids_are_none(self, unit_meta_conn):
        state = make_state(player_hero_id=None, opponent_hero_id=None)
        assert self.predictor._hero_advantage(state, unit_meta_conn) == 0.0

    def test_positive_when_player_hero_scores_higher(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "paladin", 8.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",  4.0, entity_type="Hero")
        state = make_state(player_hero_id="paladin", opponent_hero_id="shaman")
        adv = self.predictor._hero_advantage(state, unit_meta_conn)
        assert adv > 0.0

    def test_negative_when_opponent_hero_scores_higher(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "paladin", 3.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",  9.0, entity_type="Hero")
        state = make_state(player_hero_id="paladin", opponent_hero_id="shaman")
        adv = self.predictor._hero_advantage(state, unit_meta_conn)
        assert adv < 0.0

    def test_zero_when_equal_scores(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "paladin", 6.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",  6.0, entity_type="Hero")
        state = make_state(player_hero_id="paladin", opponent_hero_id="shaman")
        adv = self.predictor._hero_advantage(state, unit_meta_conn)
        assert adv == pytest.approx(0.0)

    def test_only_player_hero_known(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "paladin", 8.0, entity_type="Hero")
        state = make_state(player_hero_id="paladin", opponent_hero_id="unknown")
        adv = self.predictor._hero_advantage(state, unit_meta_conn)
        # Player score 8.0, opponent treated as 0.0 → (8-0)/10 = 0.8
        assert adv == pytest.approx(0.8)

    def test_score_normalised_to_max_score(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "paladin", 10.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",   0.0, entity_type="Hero")
        state = make_state(player_hero_id="paladin", opponent_hero_id="shaman")
        adv = self.predictor._hero_advantage(state, unit_meta_conn)
        assert adv == pytest.approx(1.0)


# ===========================================================================
# Deck tier advantage component
# ===========================================================================

class TestDeckTierAdvantage:

    def setup_method(self):
        self.predictor = WinPredictor()

    def test_returns_zero_without_db(self, populated_state):
        assert self.predictor._deck_tier_advantage(populated_state, None) == 0.0

    def test_returns_zero_when_no_scores_in_db(self, unit_meta_conn, populated_state):
        # DB has no tier_scores rows
        assert self.predictor._deck_tier_advantage(populated_state, unit_meta_conn) == 0.0

    def test_positive_when_player_deck_higher(self, unit_meta_conn):
        p_board = make_board((0, 0, make_cell("archer")))
        o_board = make_board((0, 0, make_cell("rogue")))
        insert_tier_score(unit_meta_conn, "archer", 8.0)
        insert_tier_score(unit_meta_conn, "rogue",  3.0)
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._deck_tier_advantage(state, unit_meta_conn)
        assert adv > 0.0

    def test_negative_when_opponent_deck_higher(self, unit_meta_conn):
        p_board = make_board((0, 0, make_cell("archer")))
        o_board = make_board((0, 0, make_cell("rogue")))
        insert_tier_score(unit_meta_conn, "archer", 3.0)
        insert_tier_score(unit_meta_conn, "rogue",  8.0)
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._deck_tier_advantage(state, unit_meta_conn)
        assert adv < 0.0

    def test_zero_when_equal_scores(self, unit_meta_conn):
        p_board = make_board((0, 0, make_cell("archer")))
        o_board = make_board((0, 0, make_cell("rogue")))
        insert_tier_score(unit_meta_conn, "archer", 6.0)
        insert_tier_score(unit_meta_conn, "rogue",  6.0)
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._deck_tier_advantage(state, unit_meta_conn)
        assert adv == pytest.approx(0.0)

    def test_talent_build_tag_preferred_over_generic(self, unit_meta_conn):
        # Generic score = 5.0, build-specific score = 9.0
        insert_tier_score(unit_meta_conn, "archer", 5.0, entity_build="ALL (max level)")
        insert_tier_score(unit_meta_conn, "archer", 9.0, entity_build="T3_L")
        # Cell with fully resolved T3_L path should pick up the build-specific score
        cell = make_cell("archer", talent_path={1: "L", 2: "L", 3: "L"})
        o_cell = make_cell("rogue")
        insert_tier_score(unit_meta_conn, "rogue", 5.0)
        p_board = make_board((0, 0, cell))
        o_board = make_board((0, 0, o_cell))
        state = make_state(player_board=p_board, opponent_board=o_board)
        adv = self.predictor._deck_tier_advantage(state, unit_meta_conn)
        # Player score = 9.0, opponent = 5.0 → (9-5)/10 = 0.4
        assert adv == pytest.approx(0.4)


# ===========================================================================
# Full predict() integration
# ===========================================================================

class TestPredict:

    def setup_method(self):
        self.predictor = WinPredictor()

    def test_predict_returns_float_in_unit_interval(self, empty_state):
        prob = self.predictor.predict(empty_state)
        assert 0.0 <= prob <= 1.0

    def test_predict_stores_on_state(self, empty_state):
        prob = self.predictor.predict(empty_state)
        assert empty_state.win_probability == pytest.approx(prob)

    def test_predict_no_db_no_hp_returns_near_half(self):
        # Neither rank nor HP data → only rank efficiency contributes,
        # but both boards are empty → 0 rank sum → raw = 0 → prob = 0.5
        state = make_state(player_hp=None, opponent_hp=None)
        prob = self.predictor.predict(state, db_conn=None)
        assert prob == pytest.approx(0.5)

    def test_predict_player_dominates_gives_above_half(self, unit_meta_conn):
        # Player: high-rank units + high-tier heroes + high deck scores
        # Opponent: low everything
        insert_tier_score(unit_meta_conn, "archer", 9.0)
        insert_tier_score(unit_meta_conn, "knight", 8.0)
        insert_tier_score(unit_meta_conn, "rogue",  2.0)
        insert_tier_score(unit_meta_conn, "paladin", 9.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",  2.0, entity_type="Hero")

        p_board = make_board(
            (0, 0, make_cell("archer", merge_rank=7)),
            (0, 1, make_cell("knight", merge_rank=6)),
        )
        o_board = make_board(
            (0, 0, make_cell("rogue", merge_rank=1)),
        )
        state = make_state(
            player_board=p_board,
            opponent_board=o_board,
            player_hero_id="paladin",
            opponent_hero_id="shaman",
            player_hp=3,
            opponent_hp=1,
        )
        prob = self.predictor.predict(state, db_conn=unit_meta_conn)
        assert prob > 0.65

    def test_predict_opponent_dominates_gives_below_half(self, unit_meta_conn):
        insert_tier_score(unit_meta_conn, "archer", 2.0)
        insert_tier_score(unit_meta_conn, "rogue",  9.0)
        insert_tier_score(unit_meta_conn, "paladin", 2.0, entity_type="Hero")
        insert_tier_score(unit_meta_conn, "shaman",  9.0, entity_type="Hero")

        p_board = make_board((0, 0, make_cell("archer", merge_rank=1)))
        o_board = make_board(
            (0, 0, make_cell("rogue", merge_rank=7)),
            (0, 1, make_cell("rogue", merge_rank=6)),
        )
        state = make_state(
            player_board=p_board,
            opponent_board=o_board,
            player_hero_id="paladin",
            opponent_hero_id="shaman",
            player_hp=1,
            opponent_hp=3,
        )
        prob = self.predictor.predict(state, db_conn=unit_meta_conn)
        assert prob < 0.35

    def test_predict_symmetric_situation_near_half(self, unit_meta_conn):
        # Mirror boards and equal scores → should be near 0.5
        insert_tier_score(unit_meta_conn, "archer", 5.0)
        insert_tier_score(unit_meta_conn, "rogue",  5.0)

        p_board = make_board((0, 0, make_cell("archer", merge_rank=3)))
        o_board = make_board((0, 0, make_cell("rogue",  merge_rank=3)))
        state = make_state(
            player_board=p_board,
            opponent_board=o_board,
            player_hp=3,
            opponent_hp=3,
        )
        prob = self.predictor.predict(state, db_conn=unit_meta_conn)
        assert prob == pytest.approx(0.5, abs=0.05)
