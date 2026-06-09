"""
Build a board-sprite reference library from actual gameplay video.

Cross-frame NCC tests confirmed that NCC scoring works well at this video quality
(scores of 0.66-0.94 when the same unit occupies the same cell across frames).
The only problem was the existing reference images are card portraits from the
collection screen, while the board shows a different visual style (disc sprites).

This tool:
  1. Identifies the most stable cells (unit didn't move/merge between frames)
     using cross-frame NCC scores.
  2. Saves the best-quality crop for each stable cell to data/to_label/.
  3. Generates labels.json — a JSON file you fill in with unit names.
  4. In --apply mode, copies labeled crops to assets/reference/<unit_id>/board_rank1.png.

Usage:
    # Step 1: Extract stable cells
    python tools/build_reference_from_video.py data/screenshots/hud_frames/gameplay.mp4

    # Step 2: Open data/to_label/ and labels.json.
    #         For each crop_<key>.png, identify the unit name and fill in labels.json.
    #         Compare to existing reference images in assets/reference/ for help.

    # Step 3: Apply labels to build the reference library
    python tools/build_reference_from_video.py data/screenshots/hud_frames/gameplay.mp4 --apply
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from capture.video_capture import VideoCapture
from capture.grid_calibrator import GridCalibrator

VIDEO_DEFAULT = "data/screenshots/hud_frames/gameplay.mp4"
OUT_DIR = Path("data/to_label")
LABELS_FILE = OUT_DIR / "labels.json"
REFERENCE_DIR = Path("assets/reference")

# Frames to sample for stability analysis (fraction of total).
SAMPLE_PCTS = [0.15, 0.25, 0.35, 0.49, 0.58, 0.66, 0.74, 0.83]

# Minimum average cross-frame NCC to consider a cell "stable".
STABILITY_THRESHOLD = 0.45

# Display scale for the labeling sheet (cells are tiny; scale up for visibility).
DISPLAY_SCALE = 6


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    th, tw = b.shape[:2]
    if a.size == 0 or b.size == 0:
        return 0.0
    resized = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(resized, b, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])


def _mean_brightness(cell: np.ndarray) -> float:
    if cell.size == 0:
        return 0.0
    return float(cv2.mean(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))[0])


def extract_stable_cells(video_path: str):
    """Step 1 & 2: find stable cells and save crops for labeling."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with VideoCapture(video_path) as cap:
        total = cap.frame_count
        fps = cap.fps
        print(f"Video: {total} frames @ {fps:.1f} fps ({cap.duration_sec:.1f}s)")
        print(f"Sampling {len(SAMPLE_PCTS)} frames...\n")

        frames: dict[float, np.ndarray] = {}
        for pct in SAMPLE_PCTS:
            pos = int(total * pct)
            cap._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, raw = cap._cap.read()
            if ok:
                frames[pct] = cap._process_frame(raw)

    if not frames:
        print("ERROR: Could not read any frames.")
        return

    pcts = sorted(frames.keys())
    fh, fw = frames[pcts[0]].shape[:2]
    cal = GridCalibrator.from_defaults(fw, fh)

    # For each cell: compute average NCC score across all frame pairs.
    stable: list[tuple[float, str, int, int, float, np.ndarray]] = []
    # (avg_ncc, player, row, col, best_brightness, best_crop)

    for player in ("player", "opponent"):
        for row in range(5):
            for col in range(3):
                crops = {p: cal.crop_cell(frames[p], player, row, col)
                         for p in pcts}

                # Skip cells that are ever empty.
                if any(_mean_brightness(c) < 15 for c in crops.values()):
                    continue

                # Compute all pairwise NCC scores.
                pair_scores = []
                for i, p1 in enumerate(pcts):
                    for p2 in pcts[i+1:]:
                        pair_scores.append(_ncc(crops[p1], crops[p2]))

                avg_ncc = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

                if avg_ncc < STABILITY_THRESHOLD:
                    continue  # Unit merged or changed too often

                # Pick the sharpest crop (highest Laplacian variance = sharpest).
                best_crop = max(crops.values(),
                                key=lambda c: cv2.Laplacian(
                                    cv2.cvtColor(c, cv2.COLOR_BGR2GRAY),
                                    cv2.CV_64F).var())

                stable.append((avg_ncc, player, row, col, avg_ncc, best_crop))

    # Sort by stability score descending.
    stable.sort(key=lambda x: -x[0])
    print(f"Found {len(stable)} stable cells (avg NCC >= {STABILITY_THRESHOLD}):\n")

    labels_template: dict[str, str] = {}
    sheet_crops: list[tuple[str, np.ndarray]] = []

    for avg_ncc, player, row, col, _, crop in stable:
        key = f"{'p' if player == 'player' else 'o'}_r{row}c{col}"
        print(f"  {key}  avg_NCC={avg_ncc:.3f}  ({crop.shape[1]}x{crop.shape[0]}px)")

        # Save the crop.
        crop_path = OUT_DIR / f"crop_{key}.png"
        # Save at native size AND scaled up for visibility.
        h, w = crop.shape[:2]
        big = cv2.resize(crop, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(crop_path), big)

        labels_template[key] = ""   # user fills this in
        sheet_crops.append((key, big))

    # Write labels JSON template.
    if not LABELS_FILE.exists():
        LABELS_FILE.write_text(json.dumps(labels_template, indent=2))
        print(f"\nLabels template written to: {LABELS_FILE}")
    else:
        print(f"\nLabels file already exists: {LABELS_FILE}  (not overwritten)")

    # Build a labeling contact sheet.
    _save_label_sheet(sheet_crops, OUT_DIR / "labeling_sheet.png")

    print(f"\nCrops saved to: {OUT_DIR}/")
    print("\n--- NEXT STEP ---")
    print("1. Open data/to_label/ and look at each crop_*.png file.")
    print("   (Each crop is scaled up 6x for visibility.)")
    print("2. Compare to assets/reference/ card portraits to identify the unit.")
    print("3. Fill in data/to_label/labels.json  e.g.:")
    print('   { "o_r0c0": "portal_mage", "o_r0c1": "demonologist", ... }')
    print("4. Run this script again with --apply to copy crops to the reference library.")


def _save_label_sheet(crops: list[tuple[str, np.ndarray]], out_path: Path):
    if not crops:
        return
    h, w = crops[0][1].shape[:2]
    cols = 6
    rows = (len(crops) + cols - 1) // cols
    pad = 8
    sheet_w = cols * (w + pad) + pad
    sheet_h = rows * (h + 40 + pad) + pad
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, (key, img) in enumerate(crops):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + 40 + pad)
        sheet[y:y + h, x:x + w] = img
        cv2.putText(sheet, key, (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.imwrite(str(out_path), sheet)
    print(f"Labeling sheet saved: {out_path}")


def apply_labels(video_path: str):
    """Step 3: copy labeled crops to assets/reference/<unit_id>/board_rank1.png."""
    if not LABELS_FILE.exists():
        print(f"ERROR: {LABELS_FILE} not found. Run without --apply first.")
        return

    labels: dict[str, str] = json.loads(LABELS_FILE.read_text())
    empty = [k for k, v in labels.items() if not v.strip()]
    if empty:
        print(f"WARNING: {len(empty)} cells still unlabeled: {empty}")

    copied = 0
    for key, unit_id in labels.items():
        if not unit_id.strip():
            continue
        src = OUT_DIR / f"crop_{key}.png"
        if not src.exists():
            print(f"  SKIP {key}: crop file not found")
            continue
        dst_dir = REFERENCE_DIR / unit_id.strip()
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "board_rank1.png"
        shutil.copy2(str(src), str(dst))
        print(f"  {key} -> {dst}")
        copied += 1

    print(f"\nCopied {copied} crops to reference library.")
    if copied:
        print("The template matcher will now find these board_rank1.png files")
        print("alongside any existing base_rank1.png card portraits.")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_DEFAULT
    apply_mode = "--apply" in sys.argv

    if apply_mode:
        apply_labels(video)
    else:
        extract_stable_cells(video)