"""
SQLite schema definitions for the three RRGA databases.

  DB 1 — visual_reference.db  : sprite paths and capture status per unit variant
  DB 2 — unit_meta.db         : game knowledge (units, talents, heroes, tier scores, etc.)
  DB 3 — match_history.db     : logged game state timelines with outcomes (LightGBM training data)
"""

# ---------------------------------------------------------------------------
# DB 1 — Visual Reference Library
# ---------------------------------------------------------------------------

VISUAL_REFERENCE_DDL = """
CREATE TABLE IF NOT EXISTS visual_reference (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT    NOT NULL,
    appearance_state    TEXT    NOT NULL,   -- 'base', 'max_level', 'reincarnation_1/2/3'
    merge_rank          INTEGER NOT NULL,   -- 1-7 (Treant caps at 4)
    variant_tag         TEXT,               -- e.g. 'moon', 'sun' for Twins; 'blue', 'red' for Enchanted Sword
    file_path           TEXT,               -- relative path under assets/reference/
    game_version        TEXT,
    captured            INTEGER NOT NULL DEFAULT 0,  -- 0 = not captured, 1 = captured
    UNIQUE(unit_id, appearance_state, merge_rank, variant_tag)
);

CREATE TABLE IF NOT EXISTS talent_icon_reference (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT    NOT NULL,
    tier                INTEGER NOT NULL,   -- 1-4
    file_path           TEXT,
    game_version        TEXT,
    captured            INTEGER NOT NULL DEFAULT 0,
    UNIQUE(unit_id, tier)
);

CREATE TABLE IF NOT EXISTS hero_portrait_reference (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    hero_id             TEXT    NOT NULL UNIQUE,
    file_path           TEXT,
    game_version        TEXT,
    captured            INTEGER NOT NULL DEFAULT 0
);
"""

# ---------------------------------------------------------------------------
# DB 2 — Unit Meta & Tier Database
# ---------------------------------------------------------------------------

UNIT_META_DDL = """
CREATE TABLE IF NOT EXISTS units (
    unit_id             TEXT    PRIMARY KEY,
    display_name        TEXT    NOT NULL,
    description         TEXT,
    rarity              TEXT,               -- 'Common', 'Rare', 'Epic', 'Legendary'
    faction             TEXT,
    primary_role        TEXT,               -- 'DPS', 'Support', 'CC', 'Mana Generator', 'Hybrid'
    secondary_role      TEXT,
    has_talents         INTEGER NOT NULL DEFAULT 0,
    talent_unlock_t1    INTEGER,            -- player level required
    talent_unlock_t2    INTEGER,
    talent_unlock_t3    INTEGER,
    talent_unlock_t4    INTEGER,
    has_reincarnation   INTEGER NOT NULL DEFAULT 0,
    displays_stat_nums  INTEGER NOT NULL DEFAULT 0,
    stat_num_count      INTEGER,
    has_board_anims     INTEGER NOT NULL DEFAULT 0,
    board_manipulation  INTEGER NOT NULL DEFAULT 0,
    max_merge_rank      INTEGER NOT NULL DEFAULT 7,
    research_status     TEXT    NOT NULL DEFAULT 'Not Started',
    last_updated        TEXT
);

CREATE TABLE IF NOT EXISTS talent_trees (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT    NOT NULL REFERENCES units(unit_id),
    tier                INTEGER NOT NULL,   -- 1-4
    branch              TEXT    NOT NULL,   -- 'L', 'R', 'Fixed'
    unlock_level        INTEGER,
    talent_name         TEXT,
    mechanical_effect   TEXT,
    observable_sigs     TEXT,               -- behavioral/visual signatures to detect this branch
    research_status     TEXT    NOT NULL DEFAULT 'Not Started',
    last_updated        TEXT,
    UNIQUE(unit_id, tier, branch)
);

CREATE TABLE IF NOT EXISTS stat_numbers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT    NOT NULL REFERENCES units(unit_id),
    talent_branch       TEXT,               -- NULL = base (no talent active); 'L', 'R', 'Fixed'
    talent_tier         INTEGER,            -- NULL = applies to any tier of this branch; 1-4
    position            TEXT,               -- 'top_left', 'bottom_right', etc.
    meaning             TEXT,
    scaling_formula     TEXT,               -- JSON: {"type":"piecewise","segments":[{"from":0,"to":50,"weight":1.0},{"from":50,"to":200,"weight":0.3}]}
    research_status     TEXT    NOT NULL DEFAULT 'Not Started'
);

CREATE TABLE IF NOT EXISTS animations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT    NOT NULL REFERENCES units(unit_id),
    animation_name      TEXT    NOT NULL,
    category            TEXT,               -- 'buff', 'debuff', 'intrinsic', 'merge', 'mana_powerup'
    trigger             TEXT,
    duration_sec        REAL,
    strength_modifier   REAL,               -- normalized value used in win prediction
    research_status     TEXT    NOT NULL DEFAULT 'Not Started'
);

CREATE TABLE IF NOT EXISTS synergies (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_a_id           TEXT    NOT NULL REFERENCES units(unit_id),
    unit_b_id           TEXT    NOT NULL REFERENCES units(unit_id),
    description         TEXT,
    strength_bonus      REAL,
    positional          INTEGER NOT NULL DEFAULT 0,  -- 1 if adjacency matters
    research_status     TEXT    NOT NULL DEFAULT 'Not Started'
);

CREATE TABLE IF NOT EXISTS heroes (
    hero_id             TEXT    PRIMARY KEY,
    display_name        TEXT    NOT NULL
);

-- One row per ability set (sets 1-5) per hero.
-- Set 5 is always "Second Wind" (passive) and is unlocked after 80 total points invested.
CREATE TABLE IF NOT EXISTS hero_ability_sets (
    hero_id                  TEXT    NOT NULL REFERENCES heroes(hero_id),
    set_number               INTEGER NOT NULL,  -- 1-5
    ability_name             TEXT    NOT NULL,
    ability_type             TEXT,              -- 'Active - Main', 'Active - Secondary', 'Passive'
    morale_cost              INTEGER,
    description              TEXT,
    unlock_points_required   INTEGER,           -- total hero points needed to unlock this set (set 5 = 80)
    observable_sigs          TEXT,              -- behavioral/visual signatures for this ability
    research_status          TEXT    NOT NULL DEFAULT 'Not Started',
    last_updated             TEXT,
    PRIMARY KEY (hero_id, set_number)
);

-- Investment sets per ability set.
-- Each investment set is a named category the player allocates points into
-- (e.g. "Generate mana", "Monster health"). Rules vary per hero:
--   - total_point_limit NULL  : no cap on investment
--   - total_point_limit set   : hard cap on total points ever invested
--   - requires_80_pts = 1     : this specific category is locked or partially locked
--                               until the player has invested 80 total hero points
--   - pre_80_point_limit set  : how many points can be invested before the 80-point
--                               threshold (only meaningful when requires_80_pts = 1)
--   - pre_80_point_limit NULL : category is fully locked until 80 points (0 allowed before)
-- Not every category has a limit, and the 80-point unlock applies only to specific
-- categories on specific heroes — it must be documented individually.
CREATE TABLE IF NOT EXISTS hero_investment_sets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    hero_id             TEXT    NOT NULL,
    set_number          INTEGER NOT NULL,      -- which ability set this investment belongs to
    investment_name     TEXT    NOT NULL,      -- e.g. "Generate mana", "Monster health"
    total_point_limit   INTEGER,               -- NULL = no cap; otherwise hard maximum
    requires_80_pts     INTEGER NOT NULL DEFAULT 0,  -- 1 if 80-point threshold affects this category
    pre_80_point_limit  INTEGER,               -- cap before 80 pts (NULL = fully locked until 80)
    description         TEXT,
    FOREIGN KEY (hero_id, set_number) REFERENCES hero_ability_sets(hero_id, set_number)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id         TEXT    PRIMARY KEY,
    display_name        TEXT    NOT NULL,
    slot                TEXT,
    passive_effect      TEXT,
    active_effect       TEXT,
    visual_signature    TEXT,
    research_status     TEXT    NOT NULL DEFAULT 'Not Started'
);

CREATE TABLE IF NOT EXISTS spells (
    spell_id            TEXT    PRIMARY KEY,
    display_name        TEXT    NOT NULL,
    trigger_condition   TEXT,
    effect_description  TEXT,
    visual_signature    TEXT,
    research_status     TEXT    NOT NULL DEFAULT 'Not Started'
);

CREATE TABLE IF NOT EXISTS tier_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id           TEXT    NOT NULL,
    entity_type         TEXT    NOT NULL,   -- 'Unit', 'Hero', 'Artifact'
    entity_build        TEXT,               -- 'ALL (max level)', 'NONE (no talents)', talent branch, etc.
    build_descriptor    TEXT,
    tier                TEXT,               -- 'S', 'A', 'B', 'C', 'D'
    score               REAL,
    level               INTEGER,
    patch_version       TEXT,
    strengths           TEXT,
    weaknesses          TEXT,
    notes               TEXT,
    research_status     TEXT    NOT NULL DEFAULT 'Not Started',
    last_updated        TEXT
);

CREATE TABLE IF NOT EXISTS patch_log (
    patch_version       TEXT    PRIMARY KEY,
    release_date        TEXT,
    units_changed       TEXT,
    heroes_changed      TEXT,
    new_content         TEXT,
    notes               TEXT
);
"""

# ---------------------------------------------------------------------------
# DB 3 — Match History & Analysis Store
# ---------------------------------------------------------------------------

MATCH_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id            TEXT    PRIMARY KEY,
    recorded_at         TEXT    NOT NULL,
    source_type         TEXT    NOT NULL,   -- 'live_capture', 'video_file'
    source_path         TEXT,
    game_version        TEXT,
    player_hero_id      TEXT,
    opponent_hero_id    TEXT,
    player_deck         TEXT,               -- JSON array of unit_ids
    opponent_deck       TEXT,
    outcome             TEXT,               -- 'win', 'loss', null if unknown
    total_waves         INTEGER,
    match_duration_sec  REAL
);

CREATE TABLE IF NOT EXISTS game_state_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            TEXT    NOT NULL REFERENCES matches(match_id),
    timestamp_sec       REAL    NOT NULL,   -- seconds into the match
    wave_number         INTEGER,
    player_hp           INTEGER,
    opponent_hp         INTEGER,
    player_mana         INTEGER,
    player_board        TEXT,               -- JSON: [{cell, unit_id, rank, talent_tier, variant}]
    opponent_board      TEXT,
    active_buffs        TEXT,               -- JSON array of active animation names per player
    win_probability     REAL,               -- Phase 1 formula output at this snapshot
    confidence          REAL                -- overall recognition confidence at this snapshot
);

CREATE TABLE IF NOT EXISTS unit_performance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            TEXT    NOT NULL REFERENCES matches(match_id),
    unit_id             TEXT    NOT NULL,
    player              TEXT    NOT NULL,   -- 'player' or 'opponent'
    max_rank_seen       INTEGER,
    talent_tier_seen    INTEGER,
    talent_branch       TEXT,               -- 'L', 'R', 'Fixed', null if unresolved
    branch_confidence   REAL
);
"""

ALL_DDL = {
    "visual_reference": VISUAL_REFERENCE_DDL,
    "unit_meta": UNIT_META_DDL,
    "match_history": MATCH_HISTORY_DDL,
}
