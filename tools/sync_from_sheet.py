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

Usage:
    python tools/sync_from_sheet.py              # sync all available CSVs
    python tools/sync_from_sheet.py --table units # sync one table only
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = REPO_ROOT / "data" / "sheet_exports"
sys.path.insert(0, str(REPO_ROOT))

from src.database.connection import unit_meta_db
from src.database.init_db import init_all
from src.database.unit_meta_repo import (
    UnitRepo, TalentRepo, StatNumberRepo, AnimationRepo,
    SynergyRepo, HeroRepo, TierScoreRepo, PatchLogRepo,
)


def load_csv(filename: str) -> list[dict]:
    path = EXPORTS_DIR / filename
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def clean(row: dict) -> dict:
    """Strip whitespace and convert empty strings to None. Skip None keys (extra CSV columns)."""
    return {k.strip(): (v.strip() if v and v.strip() else None) for k, v in row.items() if k is not None}


def to_int(val) -> int | None:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def to_float(val) -> float | None:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None


def to_bool(val) -> int:
    return 1 if str(val).strip().lower() in ("yes", "true", "1") else 0


# ---------------------------------------------------------------------------
# Per-table sync functions
# ---------------------------------------------------------------------------

def sync_units(conn):
    rows = load_csv("units_master.csv")
    if not rows:
        print("  [SKIP] units_master.csv not found")
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
    if not rows:
        print("  [SKIP] talent_trees.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID") or not r.get("Tier"):
            continue
        records.append({
            "unit_id":           r.get("Unit ID"),
            "tier":              to_int(r.get("Tier")),
            "branch":            r.get("Branch"),
            "unlock_level":      to_int(r.get("Unlock Level")),
            "talent_name":       r.get("Talent Name"),
            "mechanical_effect": r.get("Mechanical Effect"),
            "observable_sigs":   r.get("Observable Signatures"),
            "research_status":   r.get("Research Status") or "Not Started",
            "last_updated":      r.get("Last Updated"),
        })
    TalentRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} talent tree rows")
    return len(records)


def sync_stat_numbers(conn):
    rows = load_csv("stat_numbers.csv")
    if not rows:
        print("  [SKIP] stat_numbers.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID"):
            continue
        records.append({
            "unit_id":         r.get("Unit ID"),
            "position":        r.get("Number Position"),
            "meaning":         r.get("Meaning"),
            "display_color":   r.get("Color"),
            "scaling_formula": r.get("Scaling Formula"),
            "research_status": r.get("Research Status") or "Not Started",
        })
    StatNumberRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} stat number rows")
    return len(records)


def sync_animations(conn):
    rows = load_csv("animations.csv")
    if not rows:
        print("  [SKIP] animations.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit ID"):
            continue
        records.append({
            "unit_id":          r.get("Unit ID"),
            "animation_name":   r.get("Animation Name"),
            "category":         r.get("Category"),
            "trigger":          r.get("Trigger"),
            "duration_sec":     to_float(r.get("Duration")),
            "strength_modifier":to_float(r.get("Strength Modifier")),
            "research_status":  r.get("Research Status") or "Not Started",
        })
    AnimationRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} animation rows")
    return len(records)


def sync_synergies(conn):
    rows = load_csv("synergies.csv")
    if not rows:
        print("  [SKIP] synergies.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Unit A ID") or not r.get("Unit B ID"):
            continue
        records.append({
            "unit_a_id":      r.get("Unit A ID"),
            "unit_b_id":      r.get("Unit B ID"),
            "description":    r.get("Synergy Description"),
            "strength_bonus": to_float(r.get("Strength Bonus")),
            "positional":     to_bool(r.get("Positional")),
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
      Stat 1 Name | Stat 1 Initial Limit | Stat 1 Total Limit |
      Stat 2 Name | Stat 2 Initial Limit | Stat 2 Total Limit |
      Stat 3 Name | Stat 3 Initial Limit | Stat 3 Total Limit |
      Observable Signatures | Research Status | Last Updated

    Stat columns repeat for however many investable stats each set has.
    Limits: initial_point_limit = max before 80 total points invested;
            total_point_limit   = max after 80 total points invested.
    If a stat's two limits are equal, the full cap is available from the start.
    """
    rows = load_csv("heroes.csv")
    if not rows:
        print("  [SKIP] heroes.csv not found")
        return 0

    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Hero ID"):
            continue

        # Build investment sets list from repeating Stat N columns in the sheet.
        # Each "Stat N" in the sheet is an investment set — a named category the player
        # allocates points into. Limits and 80-point unlock rules vary per hero/category.
        investment_sets = []
        for i in range(1, 10):  # support up to 9 investment sets per ability set
            name = r.get(f"Stat {i} Name")
            if not name:
                break
            pre_80 = r.get(f"Stat {i} Initial Limit")
            total  = r.get(f"Stat {i} Total Limit")
            # requires_80_pts is true when a pre-80 limit exists and differs from total
            requires_80 = 1 if (pre_80 is not None and pre_80 != total) else 0
            investment_sets.append({
                "investment_name":   name,
                "total_point_limit": to_int(total),      # NULL = no cap
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
    if not rows:
        print("  [SKIP] tier_scores.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Entity ID"):
            continue
        records.append({
            "entity_id":        r.get("Entity ID"),
            "entity_type":      r.get("Entity Type") or "Unit",
            "entity_build":     r.get("Entity Build"),
            "build_descriptor": r.get("Build Level Descriptor"),
            "tier":             r.get("Tier"),
            "score":            to_float(r.get("Score")),
            "level":            to_int(r.get("Level")),
            "patch_version":    r.get("Patch Version"),
            "strengths":        r.get("Strengths"),
            "weaknesses":       r.get("Weaknesses"),
            "notes":            r.get("Notes"),
            "research_status":  r.get("Research Status") or "Not Started",
            "last_updated":     r.get("Last Updated"),
        })
    TierScoreRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} tier score rows")
    return len(records)


def sync_patch_log(conn):
    rows = load_csv("patch_log.csv")
    if not rows:
        print("  [SKIP] patch_log.csv not found")
        return 0
    records = []
    for r in rows:
        r = clean(r)
        if not r.get("Patch Version"):
            continue
        records.append({
            "patch_version":  r.get("Patch Version"),
            "release_date":   r.get("Date") or r.get("Release Date"),
            "units_changed":  r.get("Units Changed"),
            "heroes_changed": r.get("Heroes Changed"),
            "new_content":    r.get("New Content"),
            "notes":          r.get("Notes"),
        })
    PatchLogRepo.upsert_many(conn, records)
    print(f"  Synced {len(records)} patch log rows")
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

    print(f"\nSyncing {len(tables)} table(s) from {EXPORTS_DIR}\n")
    total = 0
    with unit_meta_db() as conn:
        for table in tables:
            total += SYNC_TABLE[table](conn)

    print(f"\nDone. {total} total rows synced.")

    skipped = [t for t in tables
               if not (EXPORTS_DIR / f"{t}.csv").exists()
               and t not in ("stat_numbers", "animations", "synergies")]
    if skipped:
        print(f"\nTo sync skipped tables, export each tab from Google Sheets as CSV to:")
        print(f"  {EXPORTS_DIR}")


if __name__ == "__main__":
    main()
