"""
Seed unit_meta.db from CSV exports of the Google Sheet research workbook.

How to export CSVs from Google Sheets:
  1. Open the sheet at https://docs.google.com/spreadsheets/d/1Egoy1wLVy2gbliLGwly4qEdLMySg8O6JGrTplvKuDFs
  2. For each tab: File > Download > Comma Separated Values (.csv)
  3. Save each file to data/sheet_exports/ using these exact names:
       units_master.csv
       talent_trees.csv
       animations.csv
       stat_numbers.csv
       synergies.csv
       heroes.csv
       artifacts.csv
       spells.csv
       tier_scores.csv
       patch_log.csv

Alternatively, the CSVs can live directly in data/ using their original
Google Sheet export names (e.g. "Units Master.csv") — the script finds them
automatically via the fallback mapping below.

Usage:
    python tools/sync_from_sheet.py              # sync all available CSVs
    python tools/sync_from_sheet.py --table units # sync one table only
"""

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = REPO_ROOT / "data" / "sheet_exports"
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT))

from src.database.connection import unit_meta_db
from src.database.init_db import init_all
from src.database.unit_meta_repo import (
    UnitRepo, TalentRepo, StatNumberRepo, AnimationRepo,
    SynergyRepo, HeroRepo, TierScoreRepo, PatchLogRepo,
    ArtifactRepo, SpellRepo,
)

# ---------------------------------------------------------------------------
# CSV loading — handles two locations and two-row headers
# ---------------------------------------------------------------------------

# Maps the normalized export name to the actual filename under data/
# (the files exported directly from Google Sheets keep their tab names)
_DATA_FALLBACK = {
    "units_master.csv": "Units Master.csv",
    "talent_trees.csv": "Talent Trees.csv",
    "animations.csv":   "Animations.csv",
    "artifacts.csv":    "Artifacts.csv",
    "spells.csv":       "Spells.csv",
    "synergies.csv":    "Synergies.csv",
    "tier_scores.csv":  "Tier Scores.csv",
    "stat_numbers.csv": "Stat Numbers.csv",
    "patch_log.csv":    "Patch Log.csv",
}


def _is_group_label_row(row: list[str]) -> bool:
    """
    Returns True if the row is a Google Sheets group-label row (e.g. 'UNIT REFERENCE')
    rather than real column headers (e.g. 'Unit ID').
    Group label rows are all-uppercase; real headers have mixed case.
    """
    first = row[0].strip() if row else ""
    return bool(first) and first == first.upper()


def load_csv(filename: str) -> list[dict] | None:
    """
    Load a CSV export.  Tries data/sheet_exports/ first; falls back to data/
    using the display name.  Auto-detects and skips the all-caps group-label
    row that some Google Sheets exports add above the real column headers.

    Returns None if the file does not exist, [] if found but has no data rows.
    """
    path = EXPORTS_DIR / filename
    if not path.exists():
        fallback_name = _DATA_FALLBACK.get(filename)
        if fallback_name:
            path = DATA_DIR / fallback_name
    if not path.exists():
        return None

    with open(path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.reader(f))

    if not all_rows:
        return []

    # Skip the group-label row if present (e.g. 'UNIT REFERENCE,,,...')
    header_idx = 1 if _is_group_label_row(all_rows[0]) else 0
    if len(all_rows) <= header_idx:
        return []

    fieldnames = [h.strip() for h in all_rows[header_idx]]
    result = []
    for row in all_rows[header_idx + 1:]:
        # Pad short rows so every field has a value
        padded = row + [""] * max(0, len(fieldnames) - len(row))
        result.append(dict(zip(fieldnames, padded)))
    return result


def _guard(rows, filename: str) -> bool:
    """Returns True if rows has data. Prints a clear message when it doesn't."""
    if rows is None:
        print(f"  [SKIP] {filename} — file not found")
        return False
    if not rows:
        print(f"  [SKIP] {filename} — no data rows yet")
        return False
    return True


def clean(row: dict) -> dict:
    """Strip whitespace and convert empty strings to None. Skip None keys."""
    return {k.strip(): (v.strip() if v and v.strip() else None)
            for k, v in row.items() if k is not None}


def to_int(val) -> int | None:
    if not val:
        return None
    s = str(val).strip()
    try:
        return int(s)
    except ValueError:
        # Handle values like "15 (max)" — extract the leading integer
        m = re.match(r'^(-?\d+)', s)
        return int(m.group(1)) if m else None


def to_float(val) -> float | None:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None


def to_bool(val) -> int:
    return 1 if str(val).strip().lower() in ("yes", "true", "1") else 0


def combine(*vals) -> str | None:
    """Join non-empty values with ' | '; return None if all are empty."""
    parts = [v for v in vals if v]
    return " | ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Per-table sync functions
# ---------------------------------------------------------------------------

def sync_units(conn):
    rows = load_csv("units_master.csv")
    if not _guard(rows, "units_master.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID"):
            continue
        records.append({
            "unit_id":            r.get("Unit ID"),
            "display_name":       r.get("Display Name"),
            "description":        r.get("Description"),
            "rarity":             r.get("Rarity"),
            "faction":            r.get("Faction"),
            "primary_role":       r.get("Primary Role"),
            "secondary_role":     r.get("Secondary Role"),
            "has_talents":        to_bool(r.get("Has Talents?")),
            "talent_unlock_t1":   to_int(r.get("Talent Unlock Lvl (T1)")),
            "talent_unlock_t2":   to_int(r.get("Talent Unlock Lvl (T2)")),
            "talent_unlock_t3":   to_int(r.get("Talent Unlock Lvl (T3)")),
            "talent_unlock_t4":   to_int(r.get("Talent Unlock Lvl (T4)")),
            "has_reincarnation":  to_bool(r.get("Has Reincarnation?")),
            "displays_stat_nums": to_bool(r.get("Displays Stat Number(s)?")),
            "stat_num_count":     to_int(r.get("How Many Numbers?")),
            "has_board_anims":    to_bool(r.get("Has Board Animations?")),
            "board_manipulation": to_bool(r.get("Board Manipulation?")),
            "research_status":    r.get("Research Status") or "Not Started",
            "last_updated":       r.get("Last Updated"),
        })
    UnitRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} units")
    return len(records)


def sync_talents(conn):
    rows = load_csv("talent_trees.csv")
    if not _guard(rows, "talent_trees.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        unit_id = r.get("Unit ID")
        tier = to_int(r.get("Talent Tier (1-4)"))
        if not unit_id or tier is None:
            continue
        records.append({
            "unit_id":           unit_id,
            "tier":              tier,
            "branch":            r.get("Branch (A/B/Fixed)"),
            "unlock_level":      to_int(r.get("Unlock Level")),
            "talent_name":       r.get("Talent Name / Label"),
            "mechanical_effect": r.get("Mechanical Effect (full description)"),
            "observable_sigs":   r.get("Observable Animation ID(s)"),
            "research_status":   r.get("Research Status") or "Not Started",
            "last_updated":      r.get("Last Updated"),
        })
    TalentRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} talent tree rows")
    return len(records)


def sync_stat_numbers(conn):
    rows = load_csv("stat_numbers.csv")
    if not _guard(rows, "stat_numbers.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID"):
            continue
        records.append({
            "unit_id":         r.get("Unit ID"),
            "talent_branch":   r.get("Talent Branch"),   # NULL = base (no talent)
            "talent_tier":     to_int(r.get("Talent Tier")),
            "position":        r.get("Number Position"),
            "meaning":         r.get("What does this number mean?"),
            "scaling_formula": r.get("Scaling Formula / Pattern"),
            "research_status": r.get("Research Status") or "Not Started",
        })
    StatNumberRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} stat number rows")
    return len(records)


def sync_animations(conn):
    rows = load_csv("animations.csv")
    if not _guard(rows, "animations.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID"):
            continue
        records.append({
            "unit_id":           r.get("Unit ID"),
            "animation_name":    r.get("Animation Display Name"),
            "category":          r.get("Category"),
            "trigger":           r.get("Trigger Condition"),
            "duration_sec":      to_float(r.get("Duration (approx ms)")),
            "strength_modifier": to_float(r.get("Strength Modifier (formula or description)")),
            "research_status":   r.get("Research Status") or "Not Started",
        })
    AnimationRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} animation rows")
    return len(records)


def sync_synergies(conn):
    rows = load_csv("synergies.csv")
    if not _guard(rows, "synergies.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit A ID") or not r.get("Unit B ID"):
            continue
        records.append({
            "unit_a_id":      r.get("Unit A ID"),
            "unit_b_id":      r.get("Unit B ID"),
            "description":    r.get("Full Mechanical Description"),
            "strength_bonus": to_float(r.get("Strength Bonus (approx)")),
            "positional":     to_bool(r.get("Positional Requirement?")),
            "research_status":r.get("Research Status") or "Not Started",
        })
    SynergyRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} synergy rows")
    return len(records)


def sync_heroes(conn):
    """
    Heroes sheet structure (one row per ability set per hero):
      Hero ID | Hero Name | Set # | Ability Name | Type | Morale Cost | Description |
      Unlock Points Required |
      Stat 1 Name | Stat 1 Initial Limit | Stat 1 Total Limit | Stat 1 Description |
      ... repeating Stat N columns ...
      Observable Signatures | Research Status | Last Updated

    Stat columns repeat for however many investable stats each set has.
    Limits: initial_point_limit = max before 80 total points invested;
            total_point_limit   = max after 80 total points invested.
    If a stat's two limits are equal, the full cap is available from the start.
    """
    rows = load_csv("heroes.csv")
    if not _guard(rows, "heroes.csv"):
        return 0

    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Hero ID"):
            continue

        investment_sets = []
        for i in range(1, 10):
            name = r.get(f"Stat {i} Name")
            if not name:
                break
            pre_80 = r.get(f"Stat {i} Initial Limit")
            total  = r.get(f"Stat {i} Total Limit")
            requires_80 = 1 if (pre_80 is not None and pre_80 != total) else 0
            investment_sets.append({
                "investment_name":   name,
                "total_point_limit": to_int(total),
                "requires_80_pts":   requires_80,
                "pre_80_point_limit":to_int(pre_80) if requires_80 else None,
                "description":       r.get(f"Stat {i} Description"),
            })

        records.append({
            "hero_id":                r.get("Hero ID"),
            "display_name":           r.get("Hero Name") or r.get("Display Name"),
            "set_number":             to_int(r.get("Set #") or r.get("Set Number")) or 1,
            "ability_name":           r.get("Ability Name"),
            "ability_type":           r.get("Type") or r.get("Ability Type"),
            "morale_cost":            to_int(r.get("Morale Cost")),
            "description":            r.get("Description"),
            "unlock_points_required": to_int(r.get("Unlock Points Required")),
            "observable_sigs":        r.get("Observable Signatures"),
            "research_status":        r.get("Research Status") or "Not Started",
            "last_updated":           r.get("Last Updated"),
            "investment_sets":        investment_sets,
        })

    HeroRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} hero ability set rows")
    return len(records)


def sync_tier_scores(conn):
    rows = load_csv("tier_scores.csv")
    if not _guard(rows, "tier_scores.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Entity ID"):
            continue
        records.append({
            "entity_id":        r.get("Entity ID"),
            "entity_type":      r.get("Entity Type") or "Unit",
            "entity_build":     r.get("Talent Build Code"),
            "build_descriptor": r.get("Notes"),
            "tier":             r.get("Tier Letter (S/A/B/C/D)"),
            "score":            to_float(r.get("Numeric Score (0-10)")),
            "level":            to_int(r.get("Assumed Level")),
            "patch_version":    r.get("Patch Version"),
            "strengths":        r.get("Matchup Strengths"),
            "weaknesses":       r.get("Matchup Weaknesses"),
            "notes":            r.get("Justification / Notes"),
            "research_status":  r.get("Research Status") or "Not Started",
            "last_updated":     r.get("Last Updated") or r.get("Date Assessed"),
        })
    TierScoreRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} tier score rows")
    return len(records)


def sync_patch_log(conn):
    rows = load_csv("patch_log.csv")
    if not _guard(rows, "patch_log.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Patch Version"):
            continue
        records.append({
            "patch_version":  r.get("Patch Version"),
            "release_date":   r.get("Release Date") or r.get("Date"),
            "units_changed":  combine(
                r.get("Units Added"),
                r.get("Units Removed"),
                r.get("Units Rebalanced"),
            ),
            "heroes_changed": r.get("Heroes Changed"),
            "new_content":    combine(
                r.get("Units Added"),
                r.get("Artifacts/Spells Changed"),
            ),
            "notes":          r.get("Update Status") or r.get("Notes"),
        })
    PatchLogRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} patch log rows")
    return len(records)


def sync_artifacts(conn):
    rows = load_csv("artifacts.csv")
    if not _guard(rows, "artifacts.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Artifact ID"):
            continue
        records.append({
            "artifact_id":     r.get("Artifact ID"),
            "display_name":    r.get("Display Name"),
            "slot":            r.get("Equip Slot"),
            "passive_effect":  r.get("Passive Effect Description"),
            "active_effect":   r.get("Active Effect Description"),
            "visual_signature":r.get("Passive Visual (what does it look like during match?)"),
            "research_status": r.get("Research Status") or "Not Started",
        })
    ArtifactRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} artifacts")
    return len(records)


def sync_spells(conn):
    rows = load_csv("spells.csv")
    if not _guard(rows, "spells.csv"):
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Spell ID"):
            continue
        records.append({
            "spell_id":           r.get("Spell ID"),
            "display_name":       r.get("Display Name"),
            "trigger_condition":  r.get("Trigger Condition"),
            "effect_description": r.get("Full Effect Description"),
            "visual_signature":   r.get("Visual on Cast (what does casting look like?)"),
            "research_status":    r.get("Research Status") or "Not Started",
        })
    SpellRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} spells")
    return len(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SYNC_TABLE = {
    "units":        sync_units,
    "talents":      sync_talents,
    "stat_numbers": sync_stat_numbers,
    "animations":   sync_animations,
    "synergies":    sync_synergies,
    "heroes":       sync_heroes,
    "tier_scores":  sync_tier_scores,
    "patch_log":    sync_patch_log,
    "artifacts":    sync_artifacts,
    "spells":       sync_spells,
}


def main():
    parser = argparse.ArgumentParser(
        description="Seed unit_meta.db from Google Sheet CSV exports"
    )
    parser.add_argument(
        "--table", choices=list(SYNC_TABLE.keys()),
        help="Sync only this table (default: all)"
    )
    args = parser.parse_args()

    print("Ensuring databases are initialized...")
    init_all()

    tables = [args.table] if args.table else list(SYNC_TABLE.keys())

    print(f"\nSyncing {len(tables)} table(s)\n")
    total = 0
    with unit_meta_db() as conn:
        for table in tables:
            total += SYNC_TABLE[table](conn)

    print(f"\nDone. {total} total rows synced.")


if __name__ == "__main__":
    main()
