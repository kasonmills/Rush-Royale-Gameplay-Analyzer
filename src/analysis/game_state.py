"""
GameState — snapshot of a single frame's worth of match data.

Populated by the capture + recognition pipeline and consumed by the
analysis engine and win predictor.

Board layout:
  Rush Royale PvP is played in portrait orientation.
  Each player has a 3-column × 5-row grid (15 cells).
  Cells are referenced as (row, col) where row=0 is the back row
  (furthest from the monster path) and row=4 is the front row.

Cell contents:
  A cell is None when empty, or a UnitCell when occupied.
  merge_rank ranges 1–7 (Treant caps at 4).

  Talents are cumulative — a unit at tier 3 also has tiers 1 and 2 active.
  talent_path maps each active tier to its chosen branch (or None if the
  branch hasn't been observed yet this match):
    {}             → no talent equipped
    {1: 'R'}       → T1 active, Right branch chosen
    {1: 'R', 2: 'Fixed', 3: 'L'}  → T3 active, full path known
    {1: None, 2: None, 3: 'L'}    → T3 badge seen; lower branches unresolved
  Branches are 'L', 'R', or 'Fixed' (XOR per tier per unit — a tier is
  either a branching choice OR fixed, never both).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UnitCell:
    """Contents of a single board cell."""
    unit_id: str                          # matches units.unit_id in unit_meta.db
    merge_rank: int                       # 1–7
    # Cumulative talent path: {tier: branch} for all active tiers.
    # branch values: 'L', 'R', 'Fixed', or None (observed but not yet resolved).
    talent_path: dict[int, Optional[str]] = field(default_factory=dict)
    appearance_state: str = "base"        # 'base', 'max_level', 'reincarnation_1/2/3'
    variant_tag: Optional[str] = None    # e.g. 'moon'/'sun' for Twins
    recognition_confidence: float = 0.0  # template match or classifier score

    @property
    def highest_talent_tier(self) -> Optional[int]:
        """The highest active talent tier, or None if no talents are equipped."""
        return max(self.talent_path.keys()) if self.talent_path else None

    @property
    def branch_confidence(self) -> float:
        """Fraction of active tiers whose branch is resolved (not None)."""
        if not self.talent_path:
            return 1.0
        resolved = sum(1 for b in self.talent_path.values() if b is not None)
        return resolved / len(self.talent_path)


@dataclass
class BoardState:
    """
    A 3×5 grid of cells for one player.
    Access via board[row][col]; both indices are 0-based.
    Row 0 = back row, row 4 = front row (nearest monster path).
    Col 0 = left, col 2 = right.
    """
    cells: list[list[Optional[UnitCell]]] = field(
        default_factory=lambda: [[None] * 3 for _ in range(5)]
    )

    def get(self, row: int, col: int) -> Optional[UnitCell]:
        return self.cells[row][col]

    def set(self, row: int, col: int, cell: Optional[UnitCell]):
        self.cells[row][col] = cell

    def occupied(self) -> list[tuple[int, int, UnitCell]]:
        """Returns list of (row, col, UnitCell) for all non-empty cells."""
        return [
            (r, c, self.cells[r][c])
            for r in range(5)
            for c in range(3)
            if self.cells[r][c] is not None
        ]

    def unit_ids(self) -> set[str]:
        """Returns the set of unit_ids present on this board."""
        return {cell.unit_id for _, _, cell in self.occupied()}


@dataclass
class GameState:
    """
    Complete snapshot of match state at a given frame/timestamp.

    pipeline_confidence is the overall recognition confidence for this frame
    (average of all cell recognition_confidence values). Frames below a
    configurable threshold should be flagged rather than written to the DB.
    """
    # Timing
    timestamp_sec: float = 0.0        # seconds into the match / video
    wave_number: Optional[int] = None

    # Health / resources
    player_hp: Optional[int] = None
    opponent_hp: Optional[int] = None
    player_mana: Optional[int] = None

    # Boards
    player_board: BoardState = field(default_factory=BoardState)
    opponent_board: BoardState = field(default_factory=BoardState)

    # Hero identities (resolved by HeroClassifier)
    player_hero_id: Optional[str] = None
    opponent_hero_id: Optional[str] = None

    # Active animations / buffs (unit_id → list of animation names)
    active_buffs: dict[str, list[str]] = field(default_factory=dict)

    # Analysis outputs
    win_probability: Optional[float] = None  # Phase 1 formula output
    pipeline_confidence: float = 0.0         # overall frame recognition quality

    # Source metadata
    source_frame_index: Optional[int] = None  # frame number in video / capture
    match_id: Optional[str] = None

    def to_snapshot_dict(self) -> dict:
        """
        Serialize to the flat dict expected by SnapshotRepo.insert().
        Boards are serialized as JSON arrays of cell dicts.
        """
        import json

        def serialize_board(board: BoardState) -> str:
            cells = []
            for row, col, cell in board.occupied():
                cells.append({
                    "cell": f"{row},{col}",
                    "unit_id": cell.unit_id,
                    "rank": cell.merge_rank,
                    "talent_tier": cell.highest_talent_tier,
                    "talent_path": cell.talent_path,
                    "variant": cell.variant_tag,
                    "confidence": round(cell.recognition_confidence, 3),
                })
            return json.dumps(cells)

        return {
            "match_id": self.match_id,
            "timestamp_sec": self.timestamp_sec,
            "wave_number": self.wave_number,
            "player_hp": self.player_hp,
            "opponent_hp": self.opponent_hp,
            "player_mana": self.player_mana,
            "player_board": serialize_board(self.player_board),
            "opponent_board": serialize_board(self.opponent_board),
            "active_buffs": json.dumps(self.active_buffs),
            "win_probability": self.win_probability,
            "confidence": round(self.pipeline_confidence, 3),
        }