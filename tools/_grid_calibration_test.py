"""
Validates grid calibration and template matching on gameplay.mp4.

Outputs to data/screenshots/hud_frames/calibration_test/:
  grid_<pct>pct.png          — frame with board grid + deck strip overlaid
  deck_<player>_<i>_<pct>pct.png  — each deck icon crop
  cell_player_r2c1_<pct>pct.png   — a mid-board cell crop (sanity check)

Run from project root:
    python tools/_grid_calibration_test.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.capture.grid_calibrator import GridCalibrator
from src.capture.video_capture import VideoCapture
from src.recognition.template_matcher import TemplateMatcher

VIDEO   = PROJECT_ROOT / "data" / "screenshots" / "hud_frames" / "gameplay.mp4"
OUTDIR  = PROJECT_ROOT / "data" / "screenshots" / "hud_frames" / "calibration_test"
REF_DIR = PROJECT_ROOT / "assets" / "reference"

OUTDIR.mkdir(exist_ok=True)

with VideoCapture(str(VIDEO), target_width=1080) as cap:
    rw, rh = cap.raw_frame_size
    print(f"Raw video: {rw}×{rh}, {cap.fps:.0f} fps, {cap.duration_sec:.1f}s")

    region = cap.detect_game_region()
    if region:
        print(f"Auto-detected game region: x={region[0]} y={region[1]} "
              f"w={region[2]} h={region[3]}  aspect={region[2]/region[3]:.2f}")
    else:
        print("Auto-detection failed — using full frame.")

    matcher = TemplateMatcher()
    if REF_DIR.is_dir():
        matcher.load_library(REF_DIR)
        print(f"Loaded templates for {len(matcher.loaded_unit_ids())} units: "
              f"{sorted(matcher.loaded_unit_ids())}")
    else:
        print(f"WARNING: reference dir not found at {REF_DIR}")

    for pct in (0.25, 0.50, 0.75):
        ts = cap.duration_sec * pct
        frame = cap.frame_at(ts)
        if frame is None:
            print(f"  [{pct*100:.0f}%] could not read frame — skipping")
            continue

        fh, fw = frame.shape[:2]
        print(f"\n=== {pct*100:.0f}% (t={ts:.1f}s)  processed frame: {fw}×{fh} ===")

        cal = GridCalibrator.from_defaults(fw, fh)

        # Grid overlay
        annotated = cal.draw_grid(frame)
        grid_path = OUTDIR / f"grid_{int(pct*100)}pct.png"
        cv2.imwrite(str(grid_path), annotated)
        print(f"  Saved {grid_path.name}")

        # Deck icon crops + template matching
        for player in ("player", "opponent"):
            icons = cal.crop_deck_icons(frame, player)
            for i, icon in enumerate(icons):
                out = OUTDIR / f"deck_{player}_{i}_{int(pct*100)}pct.png"
                cv2.imwrite(str(out), icon)

            if matcher.loaded_unit_ids():
                deck = matcher.identify_deck(icons)
                print(f"  {player:8s} deck: {sorted(deck) if deck else '(nothing above threshold)'}")
            else:
                print(f"  {player:8s} deck: (no templates loaded)")

        # One cell crop for visual sanity check
        cell = cal.crop_cell(frame, "player", row=2, col=1)
        cell_path = OUTDIR / f"cell_player_r2c1_{int(pct*100)}pct.png"
        cv2.imwrite(str(cell_path), cell)
        print(f"  Cell (player r2,c1): {cell.shape[1]}x{cell.shape[0]}px -> {cell_path.name}")

print(f"\nDone. All output in {OUTDIR}")