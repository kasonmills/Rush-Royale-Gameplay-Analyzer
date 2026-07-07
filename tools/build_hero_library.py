"""
Build the hero portrait reference library from data/screenshots/.

Copies hero PNGs to:
    assets/reference/hero_portraits/<hero_id>.png

Also detects any unit images that landed in data/screenshots/ instead of
data/screenshots/units/, moves them there, and adds them to the unit
reference library (assets/reference/<unit_id>/base_rank1.png).

Run from the project root:
    python tools/build_hero_library.py
"""

import sqlite3
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
UNITS_DIR       = SCREENSHOTS_DIR / "units"
HERO_REF_DIR    = PROJECT_ROOT / "assets" / "reference" / "hero_portraits"
UNIT_REF_DIR    = PROJECT_ROOT / "assets" / "reference"
DB_PATH         = PROJECT_ROOT / "data" / "sheet_exports" / "unit_meta.db"
UNITS_CSV       = PROJECT_ROOT / "data" / "sheet_exports" / "Units Master.csv"


def load_hero_ids() -> dict[str, str]:
    """Returns {display_name_lower: hero_id} from unit_meta.db."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT hero_id, display_name FROM heroes").fetchall()
    conn.close()
    return {display.lower(): hero_id for hero_id, display in rows}


def load_unit_ids() -> dict[str, str]:
    """Returns {display_name_lower: unit_id} from Units Master.csv."""
    import csv
    mapping: dict[str, str] = {}
    with open(UNITS_CSV, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i < 2 or len(row) < 2:
                continue
            unit_id, display = row[0].strip(), row[1].strip()
            if unit_id and display:
                mapping[display.lower()] = unit_id
    return mapping


def build():
    hero_ids = load_hero_ids()
    unit_ids = load_unit_ids()

    HERO_REF_DIR.mkdir(parents=True, exist_ok=True)
    UNITS_DIR.mkdir(parents=True, exist_ok=True)

    # Only PNGs directly in data/screenshots/ (not in subfolders)
    candidates = [p for p in SCREENSHOTS_DIR.glob("*.png")
                  if not p.name.startswith(("Balance", "Dark", "Screenshot"))]

    heroes_copied, units_moved = 0, 0

    for src in sorted(candidates):
        stem_lower = src.stem.lower()

        if stem_lower in hero_ids:
            hero_id = hero_ids[stem_lower]
            dest = HERO_REF_DIR / f"{hero_id}.png"
            img = Image.open(src).convert("RGB")
            img.save(dest, "PNG")
            print(f"  [hero] {src.name:30s} -> hero_portraits/{hero_id}.png")
            heroes_copied += 1

        elif stem_lower in unit_ids:
            unit_id = unit_ids[stem_lower]
            # Move to data/screenshots/units/
            units_dest = UNITS_DIR / src.name
            shutil.copy2(src, units_dest)
            # Also write to reference library
            ref_dir = UNIT_REF_DIR / unit_id
            ref_dir.mkdir(exist_ok=True)
            img = Image.open(src).convert("RGB")
            img.save(ref_dir / "base_rank1.png", "PNG")
            print(f"  [unit] {src.name:30s} -> {unit_id}/base_rank1.png  (also copied to units/)")
            units_moved += 1

        else:
            print(f"  [skip] {src.name} — no matching hero or unit ID")

    print(f"\nDone. {heroes_copied} hero portraits, {units_moved} unit images processed.")


if __name__ == "__main__":
    build()