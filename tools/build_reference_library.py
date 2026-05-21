"""
Build the unit reference library from data/screenshots/units/.

Reads every JPG in data/screenshots/units/, maps the filename to the correct
unit_id from Units Master.csv, converts to PNG, and writes it to:

    assets/reference/<unit_id>/base_rank1.png

Run from the project root:
    python tools/build_reference_library.py
"""

import csv
import shutil
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots" / "units"
REFERENCE_DIR   = PROJECT_ROOT / "assets" / "reference"
UNITS_CSV       = PROJECT_ROOT / "data" / "Units Master.csv"

# Manual overrides: screenshot stem (lowercase) -> unit_id
# Used for typos and names that don't match the CSV display name.
MANUAL_OVERRIDES = {
    "brusier":          "bruiser",
    "executioneer":     "executioner",
    "frankie and stein": "franky_and_stein",
}


def load_unit_id_map() -> dict[str, str]:
    """Returns {display_name_lower: unit_id} from Units Master.csv."""
    mapping: dict[str, str] = {}
    with open(UNITS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 2 or len(row) < 2:   # skip two header rows
                continue
            unit_id, display_name = row[0].strip(), row[1].strip()
            if unit_id and display_name:
                mapping[display_name.lower()] = unit_id
    return mapping


def build_library():
    name_to_id = load_unit_id_map()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(SCREENSHOTS_DIR.glob("*.jpg")) + sorted(SCREENSHOTS_DIR.glob("*.png"))
    if not sources:
        print(f"No images found in {SCREENSHOTS_DIR}")
        sys.exit(1)

    copied, skipped = 0, []

    for src in sources:
        stem_lower = src.stem.lower()

        # 1. check manual overrides first
        unit_id = MANUAL_OVERRIDES.get(stem_lower)

        # 2. fall back to CSV display-name lookup
        if unit_id is None:
            unit_id = name_to_id.get(stem_lower)

        if unit_id is None:
            skipped.append(src.name)
            continue

        dest_dir = REFERENCE_DIR / unit_id
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / "base_rank1.png"

        img = Image.open(src).convert("RGB")
        img.save(dest, "PNG")
        print(f"  {src.name:35s} -> {unit_id}/base_rank1.png")
        copied += 1

    print(f"\nDone. {copied} images written to {REFERENCE_DIR}")
    if skipped:
        print(f"\nSkipped ({len(skipped)} — no matching unit_id found):")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    build_library()