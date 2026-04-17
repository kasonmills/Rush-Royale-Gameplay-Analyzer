"""
Tests for ActivityMonitor (inside match_runner.py).

ActivityMonitor detects when a match has ended via three independent signals:

  1. idle_empty_board  — player board has had zero units for N continuous seconds
  2. idle_no_activity  — board fingerprint AND game signals (wave/HP/buffs)
                         have both been silent for N continuous seconds
  3. match_end_hp      — either player's HP reaches 0 (fires immediately)

HP values are in range 0–3 (each player starts with 3 lives in PvP).
"""

import pytest

from src.analysis.match_runner import ActivityMonitor
from tests.conftest import make_board, make_cell, make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monitor(empty_sec: float = 5.0, no_act_sec: float = 10.0) -> ActivityMonitor:
    return ActivityMonitor(empty_board_sec=empty_sec, no_activity_sec=no_act_sec)


# ===========================================================================
# HP-zero check (fires immediately, highest priority)
# ===========================================================================

class TestHPZeroCheck:

    def test_player_hp_zero_triggers_match_end(self):
        mon = make_monitor()
        state = make_state(player_hp=0, opponent_hp=3)
        result = mon.update(state, ts=0.0)
        assert result == "match_end_hp"

    def test_opponent_hp_zero_triggers_match_end(self):
        mon = make_monitor()
        state = make_state(player_hp=3, opponent_hp=0)
        result = mon.update(state, ts=0.0)
        assert result == "match_end_hp"

    def test_both_hp_zero_triggers_match_end(self):
        mon = make_monitor()
        state = make_state(player_hp=0, opponent_hp=0)
        result = mon.update(state, ts=0.0)
        assert result == "match_end_hp"

    def test_hp_zero_fires_before_other_checks(self):
        # Even with an empty board (which would also trigger idle_empty_board),
        # match_end_hp should fire first because HP=0 is checked first.
        mon = make_monitor(empty_sec=0.0)  # empty threshold already met
        state = make_state(player_hp=0, opponent_hp=3)
        state.player_board  # already empty by default
        result = mon.update(state, ts=999.0)
        assert result == "match_end_hp"

    def test_hp_none_does_not_trigger(self):
        mon = make_monitor()
        state = make_state(player_hp=None, opponent_hp=None)
        result = mon.update(state, ts=0.0)
        assert result == "active"

    def test_hp_one_does_not_trigger(self):
        mon = make_monitor()
        state = make_state(player_hp=1, opponent_hp=1)
        result = mon.update(state, ts=0.0)
        assert result == "active"


# ===========================================================================
# Empty-board check
# ===========================================================================

class TestEmptyBoardCheck:

    def test_occupied_board_never_triggers_empty(self):
        mon = make_monitor(empty_sec=5.0)
        board = make_board((0, 0, make_cell()))
        state = make_state(player_board=board)
        result = mon.update(state, ts=100.0)
        assert result == "active"

    def test_empty_board_below_threshold_stays_active(self):
        mon = make_monitor(empty_sec=5.0)
        state = make_state()  # empty board
        mon.update(state, ts=0.0)  # start timer at t=0
        result = mon.update(state, ts=4.9)
        assert result == "active"

    def test_empty_board_at_threshold_triggers(self):
        mon = make_monitor(empty_sec=5.0)
        state = make_state()  # empty board
        mon.update(state, ts=0.0)
        result = mon.update(state, ts=5.0)
        assert result == "idle_empty_board"

    def test_empty_board_beyond_threshold_triggers(self):
        mon = make_monitor(empty_sec=5.0)
        state = make_state()
        mon.update(state, ts=0.0)
        result = mon.update(state, ts=20.0)
        assert result == "idle_empty_board"

    def test_timer_resets_when_unit_appears(self):
        mon = make_monitor(empty_sec=5.0)
        empty_state = make_state()

        # Run empty for 4 seconds (not yet triggered)
        mon.update(empty_state, ts=0.0)
        mon.update(empty_state, ts=4.0)

        # Unit appears — timer should reset
        occupied_state = make_state(player_board=make_board((0, 0, make_cell())))
        mon.update(occupied_state, ts=4.5)

        # Now empty again, but from a fresh start
        mon.update(empty_state, ts=4.6)
        result = mon.update(empty_state, ts=9.0)  # 4.6 seconds since last reset, not 9
        assert result == "active"

    def test_timer_triggers_after_reset_and_second_empty_period(self):
        mon = make_monitor(empty_sec=3.0)
        empty_state = make_state()
        occupied_state = make_state(player_board=make_board((0, 0, make_cell())))

        # First empty period (not long enough)
        mon.update(empty_state, ts=0.0)
        mon.update(occupied_state, ts=2.0)  # resets timer
        mon.update(empty_state, ts=3.0)     # fresh start

        # Second empty period — 3 seconds later should trigger
        result = mon.update(empty_state, ts=6.0)
        assert result == "idle_empty_board"


# ===========================================================================
# Dual no-activity check
# ===========================================================================

class TestNoActivityCheck:

    def test_active_on_first_frame(self):
        mon = make_monitor(no_act_sec=10.0)
        state = make_state(player_board=make_board((0, 0, make_cell())))
        result = mon.update(state, ts=0.0)
        assert result == "active"

    def test_board_change_resets_activity_timer(self):
        mon = make_monitor(no_act_sec=5.0)
        board_a = make_board((0, 0, make_cell("archer", merge_rank=1)))
        board_b = make_board((0, 0, make_cell("archer", merge_rank=2)))  # rank changed → different fp

        state_a = make_state(player_board=board_a, wave_number=1)
        state_b = make_state(player_board=board_b, wave_number=1)

        mon.update(state_a, ts=0.0)
        mon.update(state_a, ts=4.0)  # 4 seconds with no change — nearly triggered
        mon.update(state_b, ts=4.1)  # board changed → resets timer
        result = mon.update(state_b, ts=8.0)  # only 3.9s since last change
        assert result == "active"

    def test_wave_change_resets_activity_timer(self):
        mon = make_monitor(no_act_sec=5.0)
        board = make_board((0, 0, make_cell()))

        state_w1 = make_state(player_board=board, wave_number=1)
        state_w2 = make_state(player_board=board, wave_number=2)

        mon.update(state_w1, ts=0.0)
        mon.update(state_w1, ts=4.5)
        mon.update(state_w2, ts=4.6)  # wave changed → resets
        result = mon.update(state_w2, ts=9.0)  # only 4.4s since wave change
        assert result == "active"

    def test_hp_change_resets_activity_timer(self):
        mon = make_monitor(no_act_sec=5.0)
        board = make_board((0, 0, make_cell()))

        state_hp3 = make_state(player_board=board, wave_number=1, player_hp=3, opponent_hp=3)
        state_hp2 = make_state(player_board=board, wave_number=1, player_hp=2, opponent_hp=3)

        mon.update(state_hp3, ts=0.0)
        mon.update(state_hp3, ts=4.5)
        mon.update(state_hp2, ts=4.6)  # player took damage → resets
        result = mon.update(state_hp2, ts=8.0)
        assert result == "active"

    def test_active_buffs_resets_activity_timer(self):
        mon = make_monitor(no_act_sec=5.0)
        board = make_board((0, 0, make_cell()))

        state_quiet = make_state(player_board=board, active_buffs={})
        state_buffed = make_state(player_board=board, active_buffs={"archer": ["fire"]})

        mon.update(state_quiet, ts=0.0)
        mon.update(state_quiet, ts=4.5)
        mon.update(state_buffed, ts=4.6)  # active buff → other_changed = True
        result = mon.update(state_quiet, ts=8.0)
        assert result == "active"

    def test_no_activity_triggers_after_threshold(self):
        mon = make_monitor(no_act_sec=5.0)
        board = make_board((0, 0, make_cell("archer", merge_rank=3)))
        state = make_state(player_board=board, wave_number=5, player_hp=2, opponent_hp=2)

        mon.update(state, ts=0.0)  # establishes last_activity at t=0
        # Same state from here on — no board/wave/HP changes
        mon.update(state, ts=3.0)
        result = mon.update(state, ts=5.1)
        assert result == "idle_no_activity"

    def test_no_activity_does_not_fire_before_first_active_frame(self):
        # Monitor should not trigger before it has seen any activity at all.
        mon = make_monitor(no_act_sec=5.0)
        state = make_state()  # empty board → no "active" frame has been established

        # Run for a long time with an empty board — but also no units,
        # so _last_activity stays at -1 until first non-empty frame.
        # However, empty board also means the empty_board check fires first —
        # so we need a non-empty board to isolate the no_activity check.
        board = make_board((0, 0, make_cell()))
        state = make_state(player_board=board)

        # First call — establishes _last_activity
        mon.update(state, ts=0.0)
        # Immediately after — cannot have triggered yet
        result = mon.update(state, ts=0.1)
        assert result == "active"

    def test_does_not_trigger_when_only_one_channel_silent(self):
        # Board unchanged but wave keeps advancing → other_changed = True → no trigger
        mon = make_monitor(no_act_sec=5.0)
        board = make_board((0, 0, make_cell("archer", merge_rank=3)))

        for wave in range(1, 20):
            state = make_state(player_board=board, wave_number=wave, player_hp=3, opponent_hp=3)
            result = mon.update(state, ts=float(wave))
            assert result == "active", f"Should be active at wave {wave}"


# ===========================================================================
# Return value contract
# ===========================================================================

class TestReturnValues:

    def test_all_return_values_are_known_strings(self):
        valid = {"active", "idle_empty_board", "idle_no_activity", "match_end_hp"}
        mon = make_monitor()

        states = [
            make_state(),  # empty board
            make_state(player_board=make_board((0, 0, make_cell()))),  # occupied
            make_state(player_hp=0),  # HP zero
        ]
        for state in states:
            result = mon.update(state, ts=0.0)
            assert result in valid, f"Unexpected return value: {result!r}"
