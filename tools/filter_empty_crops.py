"""
Delete cell crops from data/to_label/unknown/ that clearly don't contain a unit
or are already well-covered by the reference library.

Three categories are removed:

  1. NO MEDALLION  — the golden circular tile border is absent.
     Mis-calibrated crops (path decoration, wooden wall corners) that landed
     outside the actual game board.

  2. EMPTY TILE    — the golden medallion IS present but the interior is blank
     (no unit sprite).  Identified by low colour saturation inside the circle.

  3. DUPLICATE     — the crop matches an existing image in assets/reference/
     with NCC confidence >= the duplicate threshold.  We already have this unit
     covered; keeping more copies of it just adds noise to the label folder.

Usage:
    # Dry-run — prints what would be deleted, deletes nothing:
    .venv\\Scripts\\python.exe tools\\filter_empty_crops.py

    # Actually delete:
    .venv\\Scripts\\python.exe tools\\filter_empty_crops.py --delete

    # Adjust duplicate confidence threshold (default 0.75):
    .venv\\Scripts\\python.exe tools\\filter_empty_crops.py --dup-threshold 0.80
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.recognition.template_matcher import TemplateMatcher

_UNKNOWN_DIR = _ROOT / "data" / "to_label" / "unknown"
_REF_DIR     = _ROOT / "assets" / "reference"

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Mean Sobel gradient magnitude in the RING region (r = 28–52 % of half-width).
# The circular medallion border always creates strong edges here regardless of
# animation colour.  Path decoration and wooden walls have weak or anisotropic
# gradients in this region.
_RING_GRADIENT_MEAN = 18.0    # below this → no circular structure → delete

# Fraction of pixels in the CENTRE region (radius < 38 % of half-width)
# with saturation > 70 (out of 255).  Low value = empty interior (no unit).
_CENTRE_SAT_FRAC    = 0.08    # below this (circular structure present) → empty tile → delete

# NCC confidence above which a crop is considered already covered by the
# reference library.  0.75 is high enough to avoid false matches between
# visually similar units while still catching near-identical crops.
_DEFAULT_DUP_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Structural (no-unit) check
# ---------------------------------------------------------------------------

def _structural_label(img_bgr: np.ndarray) -> str:
    """
    Returns:
      'keep'         — unit appears to be present
      'no_medallion' — crop is outside the game board (no circular structure)
      'empty_tile'   — medallion present but interior is blank (no unit)
    """
    h, w = img_bgr.shape[:2]

    cy, cx = h / 2.0, w / 2.0
    hw     = min(h, w) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dist   = np.sqrt(((ys - cy) / hw) ** 2 + ((xs - cx) / hw) ** 2)

    # ── Circular structure check via gradient magnitude in ring ───────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    ring = (dist >= 0.28) & (dist <= 0.52)
    ring_grad_mean = float(grad[ring].mean()) if ring.sum() > 0 else 0.0

    if ring_grad_mean < _RING_GRADIENT_MEAN:
        return "no_medallion"

    # ── Empty-interior check via saturation in centre ─────────────────────
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S    = hsv[:, :, 1]

    centre = dist < 0.38
    centre_total = int(centre.sum())
    if centre_total == 0:
        return "keep"

    centre_sat_frac = float((S[centre] > 70).sum()) / centre_total
    if centre_sat_frac < _CENTRE_SAT_FRAC:
        return "empty_tile"

    return "keep"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Filter empty / mis-calibrated / already-covered crops "
                    "from data/to_label/unknown/."
    )
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete files (default: dry-run only)")
    ap.add_argument("--dir", default=str(_UNKNOWN_DIR),
                    help="Directory to scan (default: data/to_label/unknown)")
    ap.add_argument("--dup-threshold", type=float, default=_DEFAULT_DUP_THRESHOLD,
                    metavar="CONF",
                    help=f"NCC confidence above which a crop is treated as a "
                         f"duplicate of an existing reference "
                         f"(default: {_DEFAULT_DUP_THRESHOLD})")
    args = ap.parse_args()

    scan_dir = Path(args.dir)
    if not scan_dir.is_dir():
        print(f"ERROR: directory not found: {scan_dir}")
        sys.exit(1)

    # ── Load reference library for duplicate detection ────────────────────
    matcher = TemplateMatcher(cell_threshold=args.dup_threshold)
    has_refs = _REF_DIR.is_dir()
    if has_refs:
        matcher.load_library(_REF_DIR)
        print(f"Reference library: {len(matcher._templates)} templates loaded.")
    else:
        print("WARNING: reference directory not found — duplicate check disabled.")

    files = sorted(scan_dir.glob("*.png"))
    if not files:
        print("No PNG files found.")
        sys.exit(0)

    print(f"Scanning {len(files)} crops in {scan_dir} ...")
    print()

    keys = ("keep", "no_medallion", "empty_tile", "duplicate")
    counts = {k: 0 for k in keys}
    to_delete: list[tuple[str, Path]] = []

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue

        # Structural checks first (fast, no template matching)
        label = _structural_label(img)

        # Duplicate check only if the crop looks like it contains a unit
        if label == "keep" and has_refs:
            result = matcher.match_cell(img, unit_ids=None)
            if result.unit_id is not None and result.confidence >= args.dup_threshold:
                label = "duplicate"

        counts[label] += 1
        if label != "keep":
            to_delete.append((label, path))

    total = len(files)
    print(f"  Keep (needs labelling) : {counts['keep']:>5}  ({counts['keep']/total*100:.1f}%)")
    print(f"  No medallion (bad area): {counts['no_medallion']:>5}  ({counts['no_medallion']/total*100:.1f}%)")
    print(f"  Empty tile (no unit)   : {counts['empty_tile']:>5}  ({counts['empty_tile']/total*100:.1f}%)")
    print(f"  Duplicate (in ref lib) : {counts['duplicate']:>5}  ({counts['duplicate']/total*100:.1f}%)")
    print(f"  Total to delete        : {len(to_delete):>5}")
    print()

    if not to_delete:
        print("Nothing to delete.")
        return

    if not args.delete:
        print("DRY RUN — pass --delete to actually remove files.")
        print()
        print("Sample of files that would be deleted (first 20):")
        for label, p in to_delete[:20]:
            print(f"  [{label:<12}]  {p.name}")
        return

    deleted = 0
    for label, path in to_delete:
        try:
            path.unlink()
            deleted += 1
        except OSError as e:
            print(f"  WARNING: could not delete {path.name}: {e}")

    print(f"Deleted {deleted} files.")


if __name__ == "__main__":
    main()