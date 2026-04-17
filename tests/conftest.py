"""
Shared pytest fixtures for the Rush Royale Gameplay Analyzer test suite.

All database fixtures use in-memory SQLite connections so tests are:
  - Fast (no disk I/O)
  - Isolated (each test gets a fresh DB)
  - Portable (no dependency on data/ directory being initialised)

Board / GameState builder helpers let each test describe only the fields
it cares about and get sensible defaults for everything else.

Game rules reminder
-------------------
- Each player has 3 lives (HP range 0–3) in PvP mode.
- Board is 3 columns × 5 rows; merge ranks run 1–7 (Treant caps at 4).
- A deck is exactly 5 unit types.
"""

import json
import sqlite3

import pytest

from src.analysis.game_state import BoardState, GameState, UnitCell
from src.database.schema import (
    MATCH_HISTORY_DDL,
    SUMMON_ANALYSIS_DDL,
    UNIT_META_DDL,
)


# ---------------------------------------------------------------------------
# In-memory database fixtures
# ---------------------------------------------------------------------------

def _make_conn(ddl: str) -> sqlite3.Connection:
    """Create an in-memory SQLite connection and apply the given DDL."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(ddl)
    conn.commit()
    return conn


@pytest.fixture
def unit_meta_conn():
    """In-memory unit_meta.db with schema applied."""
    conn = _make_conn(UNIT_META_DDL)
    yield conn
    conn.close()


@pytest.fixture
def match_history_conn():
    """In-memory match_history.db with schema applied."""
    conn = _make_conn(MATCH_HISTORY_DDL)
    yield conn
    conn.close()


@pytest.fixture
def summon_analysis_conn():
    """In-memory summon_analysis.db with schema applied."""
    conn = _make_conn(SUMMON_ANALYSIS_DDL)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Tier score helpers
# ---------------------------------------------------------------------------

def insert_tier_score(conn: sqlite3.Connection,
                      entity_id: str,
                      score: float,
                      entity_type: str = "Unit",
                      entity_build: str = "ALL (max level)"):
    """Insert a single tier_scores row into the given connection."""
    conn.execute(
        """INSERT INTO tier_scores
           (entity_id, entity_type, entity_build, score, research_status)
           VALUES (?, ?, ?, ?, 'Done')""",
        (entity_id, entity_type, entity_build, score),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# UnitCell / BoardState / GameState builders
# ---------------------------------------------------------------------------

def make_cell(unit_id: str = "archer",
              merge_rank: int = 1,
              talent_path: dict | None = None,
              confidence: float = 0.9) -> UnitCell:
    """Return a UnitCell with sensible defaults."""
    return UnitCell(
        unit_id=unit_id,
        merge_rank=merge_rank,
        talent_path=talent_path or {},
        recognition_confidence=confidence,
    )


def make_board(*cells: tuple) -> BoardState:
    """
    Build a BoardState from (row, col, UnitCell) tuples.

    Example:
        board = make_board((0, 0, make_cell("archer", 3)),
                           (1, 2, make_cell("knight", 2)))
    """
    board = BoardState()
    for row, col, cell in cells:
        board.set(row, col, cell)
    return board


def make_state(**kwargs) -> GameState:
    """
    Build a GameState with sensible defaults.

    All GameState fields can be overridden via keyword arguments.
    HP is 0–3 (each PvP player starts with 3 lives).
    """
    defaults = dict(
        timestamp_sec=0.0,
        wave_number=1,
        player_hp=3,
        opponent_hp=3,
        player_mana=0,
        player_board=BoardState(),
        opponent_board=BoardState(),
        player_hero_id=None,
        opponent_hero_id=None,
        match_id="test-match-001",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


# ---------------------------------------------------------------------------
# Fixtures that expose the builders as pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_state():
    """A GameState with empty boards, full HP (3 lives each), wave 1."""
    return make_state()


@pytest.fixture
def populated_state():
    """
    A GameState with a small, realistic board population.
    Player has 3 units at various ranks; opponent has 2.
    Player is ahead on damage (opponent on 1 life vs player on 2).
    """
    p_board = make_board(
        (0, 0, make_cell("archer",  merge_rank=3)),
        (1, 1, make_cell("knight",  merge_rank=2)),
        (4, 2, make_cell("chemist", merge_rank=1)),
    )
    o_board = make_board(
        (0, 0, make_cell("rogue",   merge_rank=4)),
        (2, 1, make_cell("monk",    merge_rank=2)),
    )
    return make_state(
        player_board=p_board,
        opponent_board=o_board,
        player_hp=2,
        opponent_hp=1,
        wave_number=5,
    )
