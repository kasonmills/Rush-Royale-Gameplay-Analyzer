"""
Synergy detector — identifies active unit-pair synergies on a board.

Rush Royale has explicit synergies described in unit tooltips (e.g.
Knight Statue adjacent to Inquisitor) and additional synergies discovered
through community research.  Both categories are stored as data, not code:
add a row to Synergies.csv or the synergies DB table and it is
automatically detected — no code changes required.

Synergy types
-------------
Non-positional:
  Both units appear anywhere on the same board.

Positional (positional=True):
  The two units must be 4-directionally adjacent (share an edge on the
  3×5 grid).  Diagonal neighbours do NOT count.

Talent-dependent synergies (talent_dependent=True in the CSV) are loaded
and detected like non-positional synergies for now — talent branch
validation is deferred until talent observation data is reliable enough
to be used as a hard filter.

Usage:
    detector = SynergyDetector()
    detector.load_from_csv("data/Synergies.csv")

    results = detector.detect(state.player_board)
    # → [SynergyResult(unit_a_id='inquisitor', unit_b_id='knight_statue', ...)]
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.analysis.game_state import BoardState


# ---------------------------------------------------------------------------
# Strength bonus parsing
# ---------------------------------------------------------------------------

_STRENGTH_KEYWORDS = {"high": 1.0, "med": 0.5, "medium": 0.5, "low": 0.25}


def _parse_strength(raw: str) -> float:
    """
    Parse a strength value from CSV text.

    Accepts:
      - A numeric string ("0.8", "1.0")
      - A keyword optionally followed by qualifiers ("High (TBD exact %)")
      - Unknown text → defaults to 0.5 (medium)
    """
    raw = raw.strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        lower = raw.lower()
        for keyword, value in _STRENGTH_KEYWORDS.items():
            if keyword in lower:
                return value
        return 0.5


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("yes", "true", "1")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SynergyEntry:
    """One registered synergy pair (internal registry entry)."""
    unit_a_id: str
    unit_b_id: str
    synergy_name: str      # human-readable label (e.g. "Knight Statue Adjacency Buff")
    synergy_type: str      # e.g. "DPS Amplifier"
    description: str
    strength_bonus: float  # 0.0–1.0 win-prediction weight
    positional: bool       # True = units must be 4-directionally adjacent
    research_status: str


@dataclass
class SynergyResult:
    """An active synergy detected on a board."""
    unit_a_id: str
    unit_b_id: str
    synergy_name: str
    strength_bonus: float
    positional: bool       # True = adjacency was verified


# ---------------------------------------------------------------------------
# SynergyDetector
# ---------------------------------------------------------------------------

class SynergyDetector:
    """
    Detects active synergies on a single BoardState.

    Load the registry with load_from_csv() or load_from_db(), then call
    detect() on any BoardState. Returns only synergies whose conditions
    are met (both units present, adjacency satisfied for positional pairs).

    Adding a new synergy requires only a new row in Synergies.csv or the
    synergies DB table — no code changes.
    """

    def __init__(self):
        self._entries: list[SynergyEntry] = []

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def load_from_csv(self, path: str | Path) -> None:
        """
        Populate the registry from Synergies.csv.

        The file uses a two-row header: the first row is a group-label row
        that is skipped; the second contains the actual column names.

        Expected columns (by name):
          Unit A ID, Unit B ID, Synergy Name / Label, Synergy Type,
          Full Mechanical Description, Strength Bonus (approx),
          Positional Requirement?, Research Status
        """
        self._entries.clear()
        with open(Path(path), newline="", encoding="utf-8") as f:
            all_rows = list(csv.reader(f))

        if len(all_rows) < 2:
            return

        fieldnames = all_rows[1]       # row 1 = real column headers
        for raw in all_rows[2:]:       # row 2+ = data
            row = dict(zip(fieldnames, raw))

            unit_a = row.get("Unit A ID", "").strip()
            unit_b = row.get("Unit B ID", "").strip()
            if not unit_a or not unit_b:
                continue

            strength_raw = row.get("Strength Bonus (approx)", "").strip()

            self._entries.append(SynergyEntry(
                unit_a_id=unit_a,
                unit_b_id=unit_b,
                synergy_name=row.get("Synergy Name / Label", "").strip(),
                synergy_type=row.get("Synergy Type", "").strip(),
                description=row.get("Full Mechanical Description", "").strip(),
                strength_bonus=_parse_strength(strength_raw),
                positional=_parse_bool(row.get("Positional Requirement?", "")),
                research_status=row.get("Research Status", "Not Started").strip(),
            ))

    def load_from_db(self, conn) -> None:
        """
        Populate the registry from the synergies table in unit_meta.db.
        Clears any previously loaded entries.

        Columns available in the schema:
          unit_a_id, unit_b_id, description, strength_bonus, positional, research_status
        synergy_name and synergy_type are not stored in the DB schema, so
        a label is derived as '<unit_a_id> + <unit_b_id>'.
        """
        self._entries.clear()
        rows = conn.execute(
            """
            SELECT unit_a_id, unit_b_id, description,
                   strength_bonus, positional, research_status
            FROM synergies
            """
        ).fetchall()
        for row in rows:
            sb = row["strength_bonus"]
            try:
                strength = float(sb) if sb is not None else 0.5
            except (TypeError, ValueError):
                strength = _parse_strength(str(sb))

            self._entries.append(SynergyEntry(
                unit_a_id=row["unit_a_id"],
                unit_b_id=row["unit_b_id"],
                synergy_name=f"{row['unit_a_id']} + {row['unit_b_id']}",
                synergy_type="",
                description=row["description"] or "",
                strength_bonus=max(0.0, min(1.0, strength)),
                positional=bool(row["positional"]),
                research_status=row["research_status"] or "Not Started",
            ))

    def known_pairs(self) -> set[frozenset[str]]:
        """Return the set of unit-id pairs that have a registered synergy."""
        return {frozenset({e.unit_a_id, e.unit_b_id}) for e in self._entries}

    def entry_count(self) -> int:
        """Total number of registered synergy entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, board: BoardState) -> list[SynergyResult]:
        """
        Return all synergies whose conditions are currently met on the board.

        For each registered entry:
          1. Both units must be present on the board.
          2. If positional=True, at least one pair of cells (one from each
             unit) must be 4-directionally adjacent.

        Returns an empty list when no synergies are active.
        """
        present_ids = board.unit_ids()
        results: list[SynergyResult] = []

        for entry in self._entries:
            if entry.unit_a_id not in present_ids or entry.unit_b_id not in present_ids:
                continue

            if entry.positional and not self._are_adjacent(
                board, entry.unit_a_id, entry.unit_b_id
            ):
                continue

            results.append(SynergyResult(
                unit_a_id=entry.unit_a_id,
                unit_b_id=entry.unit_b_id,
                synergy_name=entry.synergy_name,
                strength_bonus=entry.strength_bonus,
                positional=entry.positional,
            ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _are_adjacent(self,
                      board: BoardState,
                      unit_a_id: str,
                      unit_b_id: str) -> bool:
        """
        Return True if any cell containing unit_a_id is 4-directionally
        adjacent to any cell containing unit_b_id.

        Adjacency is defined as Manhattan distance == 1 (shared edge only;
        diagonal neighbours do not count).
        """
        a_positions = [
            (r, c) for r, c, cell in board.occupied()
            if cell.unit_id == unit_a_id
        ]
        b_positions = [
            (r, c) for r, c, cell in board.occupied()
            if cell.unit_id == unit_b_id
        ]
        for ra, ca in a_positions:
            for rb, cb in b_positions:
                if abs(ra - rb) + abs(ca - cb) == 1:
                    return True
        return False