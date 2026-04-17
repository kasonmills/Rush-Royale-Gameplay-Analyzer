"""
Win probability predictor — Phase 1 weighted formula.

Produces a win probability (0.0–1.0) for the player side from a GameState
snapshot.  Values above 0.5 favour the player; below 0.5 favour the opponent.

Formula (7 components, weights summing to 97% with talent weighting as a
modifier rather than a standalone term):

  Component                         Weight   Availability
  ─────────────────────────────────────────────────────────
  Deck tier score differential       45 %    Phase 2 (requires tier_scores DB data)
  Board rank efficiency              15 %    Phase 1 (computed from GameState alone)
  Active buff / debuff modifiers     12 %    Phase 3 (animation detection)
  Unit stat number contributions     10 %    Phase 3 (stat OCR)
  Hero contribution                   8 %    Phase 2 (requires tier_scores DB data)
  Synergy activation score            5 %    Phase 3 (complex, requires animation)
  Wave survival trajectory            2 %    Phase 1 (HP values from OCR)
  ─────────────────────────────────────────────────────────
  Talent build probability weighting is applied as a modifier inside the
  deck tier score component: if a unit's talent build is known, the
  build-specific tier score entry is used instead of the generic one.

Components that are not yet implemented return 0.0 (neutral) so the formula
degrades gracefully — those weights simply have no effect on the output until
Phase 3 data is available.

Math:
  Each component returns an advantage value in [-1.0, 1.0] where:
    +1.0 = player has maximum possible advantage on this dimension
     0.0 = roughly equal
    -1.0 = opponent has maximum possible advantage

  raw_score = sum(weight_i * advantage_i)   ∈ [-1.0, 1.0]
  probability = sigmoid(raw_score * SENSITIVITY)

  SENSITIVITY = 3.0 gives:
    raw ±0.50 → probability ≈ 0.82 / 0.18  (clear advantage)
    raw ±0.25 → probability ≈ 0.68 / 0.32  (moderate edge)
    raw  0.00 → probability = 0.50          (even)

Usage:
    predictor = WinPredictor()

    # With DB data (recommended):
    with unit_meta_db() as db:
        prob = predictor.predict(game_state, db_conn=db)

    # Without DB (formula uses rank-only components):
    prob = predictor.predict(game_state)
"""

import math
from typing import Optional

from src.analysis.game_state import GameState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Component weights — must sum to ≤ 1.0
_W_DECK_TIER    = 0.45
_W_RANK_EFF     = 0.15
_W_BUFFS        = 0.12   # Phase 3 — always 0.0 until animation detection
_W_STAT_NUMS    = 0.10   # Phase 3 — always 0.0 until stat OCR
_W_HERO         = 0.08
_W_SYNERGY      = 0.05   # Phase 3 — always 0.0 until synergy detection
_W_HP_TRAJ      = 0.02

# Scales how sharply the sigmoid converts raw advantage to probability.
# Higher = more extreme probabilities for the same raw score.
_SENSITIVITY = 3.0

# Tier score normalisation.  All scores in tier_scores are expected to sit
# in [0, _MAX_SCORE].  Adjust if the actual data uses a different scale.
_MAX_SCORE = 10.0

# Each player starts with 3 lives in Rush Royale PvP.
# HP values observed by OCR are in the range 0–3.
_MAX_HP = 3


# ---------------------------------------------------------------------------
# WinPredictor
# ---------------------------------------------------------------------------

class WinPredictor:
    """
    Computes win probability from a GameState snapshot using the Phase 1
    weighted formula.

    The predictor is stateless — call predict() on any GameState.  When a
    db_conn is provided the tier-score-dependent components are populated;
    without one, only rank efficiency and HP trajectory contribute.
    """

    def predict(self,
                state: GameState,
                db_conn=None) -> float:
        """
        Compute and return win probability for the player side.

        Args:
            state:   A GameState snapshot (from MCR.process_frame).
            db_conn: Optional open sqlite3 connection to unit_meta.db.
                     Required for tier score and hero components.

        Returns:
            Float in [0.0, 1.0].  0.5 when evidence is insufficient.
        """
        deck_tier   = _W_DECK_TIER  * self._deck_tier_advantage(state, db_conn)
        rank_eff    = _W_RANK_EFF   * self._rank_efficiency_advantage(state)
        hero        = _W_HERO       * self._hero_advantage(state, db_conn)
        hp_traj     = _W_HP_TRAJ    * self._hp_trajectory_advantage(state)

        # Phase 3 components — return 0.0 until implemented
        buffs       = _W_BUFFS    * 0.0
        stat_nums   = _W_STAT_NUMS * 0.0
        synergy     = _W_SYNERGY  * 0.0

        raw = deck_tier + rank_eff + hero + hp_traj + buffs + stat_nums + synergy
        probability = _sigmoid(raw * _SENSITIVITY)

        # Store back on the state for convenience
        state.win_probability = probability
        return probability

    # ------------------------------------------------------------------
    # Component: deck tier score differential (45 %)
    # ------------------------------------------------------------------

    def _deck_tier_advantage(self,
                              state: GameState,
                              db_conn) -> float:
        """
        Returns the normalised tier score differential between the two decks.

        For each unit in a deck the score is looked up from tier_scores.
        If the unit's talent build is fully resolved, the build-specific entry
        is tried first (talent weighting); the generic 'base' entry is used as
        fallback.

        Returns 0.0 if db_conn is None or no tier data exists yet.
        """
        if db_conn is None:
            return 0.0

        player_score = self._mean_deck_score(
            state.player_board.unit_ids(),
            {(r, c, cell.unit_id): cell
             for r, c, cell in state.player_board.occupied()},
            db_conn,
        )
        opp_score = self._mean_deck_score(
            state.opponent_board.unit_ids(),
            {(r, c, cell.unit_id): cell
             for r, c, cell in state.opponent_board.occupied()},
            db_conn,
        )

        if player_score is None and opp_score is None:
            return 0.0

        p = player_score or 0.0
        o = opp_score   or 0.0
        # Normalise: difference / max possible difference (full scale both sides)
        return (p - o) / _MAX_SCORE

    def _mean_deck_score(self,
                          unit_ids: set[str],
                          cell_map: dict,
                          db_conn) -> Optional[float]:
        """
        Average tier score across a set of units, with talent build weighting.

        cell_map: {(row, col, unit_id): UnitCell} — used to get talent_path.
        Returns None if no scores are found for any unit.
        """
        scores: list[float] = []
        for unit_id in unit_ids:
            # Find the UnitCell for this unit (take first match)
            cell = next(
                (c for (_, _, uid), c in cell_map.items() if uid == unit_id),
                None
            )
            score = self._lookup_tier_score(unit_id, cell, db_conn)
            if score is not None:
                scores.append(score)

        return sum(scores) / len(scores) if scores else None

    def _lookup_tier_score(self,
                            unit_id: str,
                            cell,
                            db_conn) -> Optional[float]:
        """
        Look up the tier score for a unit, preferring build-specific entries.

        Lookup order:
          1. Talent-build-specific entry (e.g. entity_build = 'T3_L').
          2. 'ALL (max level)' generic entry.
          3. Any entry for this unit.
        Returns None if nothing is found.
        """
        # Build a descriptor from the talent path if fully resolved
        build_tag: Optional[str] = None
        if cell is not None and cell.talent_path:
            top = cell.highest_talent_tier
            branch = cell.talent_path.get(top)
            if top is not None and branch is not None:
                build_tag = f"T{top}_{branch}"

        candidates = []
        if build_tag:
            candidates.append(build_tag)
        candidates += ["ALL (max level)", None]

        for build in candidates:
            if build is None:
                row = db_conn.execute(
                    "SELECT score FROM tier_scores "
                    "WHERE entity_id = ? AND entity_type = 'Unit' "
                    "ORDER BY score DESC LIMIT 1",
                    (unit_id,)
                ).fetchone()
            else:
                row = db_conn.execute(
                    "SELECT score FROM tier_scores "
                    "WHERE entity_id = ? AND entity_type = 'Unit' "
                    "AND entity_build = ?",
                    (unit_id, build)
                ).fetchone()
            if row is not None and row["score"] is not None:
                return float(row["score"])

        return None

    # ------------------------------------------------------------------
    # Component: board rank efficiency (15 %)
    # ------------------------------------------------------------------

    def _rank_efficiency_advantage(self, state: GameState) -> float:
        """
        Compares the total invested merge rank across both boards.

        Advantage = (player_rank_sum - opp_rank_sum) / max_possible_diff.
        max_possible_diff = 5 units × 7 ranks = 35 (one side full, other empty).
        Returns 0.0 if both boards are empty.
        """
        p_sum = sum(cell.merge_rank for _, _, cell in state.player_board.occupied())
        o_sum = sum(cell.merge_rank for _, _, cell in state.opponent_board.occupied())

        if p_sum == 0 and o_sum == 0:
            return 0.0

        # Normalise by the theoretical max one-sided advantage
        max_diff = 5 * 7  # 5 units × max rank 7
        return (p_sum - o_sum) / max_diff

    # ------------------------------------------------------------------
    # Component: hero contribution (8 %)
    # ------------------------------------------------------------------

    def _hero_advantage(self,
                         state: GameState,
                         db_conn) -> float:
        """
        Compares tier scores for the identified heroes.

        Returns 0.0 if db_conn is None or neither hero has a tier score entry.
        """
        if db_conn is None:
            return 0.0

        p_score = self._lookup_hero_score(state.player_hero_id, db_conn)
        o_score = self._lookup_hero_score(state.opponent_hero_id, db_conn)

        if p_score is None and o_score is None:
            return 0.0

        p = p_score or 0.0
        o = o_score or 0.0
        return (p - o) / _MAX_SCORE

    def _lookup_hero_score(self,
                            hero_id: Optional[str],
                            db_conn) -> Optional[float]:
        if hero_id is None:
            return None
        row = db_conn.execute(
            "SELECT score FROM tier_scores "
            "WHERE entity_id = ? AND entity_type = 'Hero' "
            "ORDER BY score DESC LIMIT 1",
            (hero_id,)
        ).fetchone()
        return float(row["score"]) if row and row["score"] is not None else None

    # ------------------------------------------------------------------
    # Component: wave survival trajectory (2 %)
    # ------------------------------------------------------------------

    def _hp_trajectory_advantage(self, state: GameState) -> float:
        """
        Advantage from HP differential.

        Positive when the opponent has lower HP (player is ahead in damage).
        Returns 0.0 if HP values are unavailable.
        """
        if state.player_hp is None or state.opponent_hp is None:
            return 0.0

        return (state.player_hp - state.opponent_hp) / _MAX_HP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Logistic sigmoid, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))
