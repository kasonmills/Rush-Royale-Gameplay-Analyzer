"""
Diagnostic: for each specified cell crop, show the top-N NCC scores and
save a side-by-side comparison image so you can visually inspect why
a crop does or doesn't match its reference images.

Output goes to data/screenshots/hud_debug/match_diag/
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.capture.grid_calibrator import GridCalibrator
from src.capture.video_capture import VideoCapture
from src.recognition.template_matcher import TemplateMatcher

VIDEO     = _ROOT / "data" / "screenshots" / "hud_frames" / "VID_20260624121536.mp4"
REF_DIR   = _ROOT / "assets" / "reference"
OUT_DIR   = _ROOT / "data" / "screenshots" / "hud_debug" / "match_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N      = 5
TIMESTAMPS = [5.0, 15.0, 30.0, 50.0]
# (side, row, col) — sample cells across both boards
CELLS = [
    ("player",   0, 0), ("player",   0, 2), ("player",   0, 4),
    ("player",   1, 1), ("player",   1, 3),
    ("player",   2, 0), ("player",   2, 2), ("player",   2, 4),
    ("opponent", 0, 1), ("opponent", 1, 2),
]

TILE_H = 96   # height of each tile in the comparison strip


def _load_all_templates(ref_dir: Path):
    """Return list of (unit_id, appearance_state, merge_rank, filepath, img)."""
    entries = []
    for unit_dir in sorted(ref_dir.iterdir()):
        if not unit_dir.is_dir() or unit_dir.name in (
                "hero_portraits", "talent_icons", "artifacts", "hero_board_effects"):
            continue
        for img_path in sorted(unit_dir.rglob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            entries.append((unit_dir.name, img_path.stem, img_path, img))
    return entries


def _ncc(crop: np.ndarray, tmpl: np.ndarray) -> float:
    th, tw = tmpl.shape[:2]
    resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    result  = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[3][0] if False else cv2.minMaxLoc(result)[1])


def _best_ncc_for_entry(crop, imgs):
    return max(_ncc(crop, img) for img in imgs)


def make_comparison(crop: np.ndarray,
                    top_entries: list,
                    label: str) -> np.ndarray:
    """
    Build a horizontal strip:
      [crop | top1_ref | top2_ref | ... | topN_ref]
    Each tile is TILE_H × TILE_H with score and name below it.
    """
    pad   = 4
    label_h = 28
    tile_w  = TILE_H

    def _make_tile(img, text_lines):
        tile = cv2.resize(img, (tile_w, TILE_H), interpolation=cv2.INTER_AREA)
        canvas = np.ones((TILE_H + label_h * len(text_lines) + pad,
                          tile_w + pad * 2, 3), dtype=np.uint8) * 30
        canvas[pad:pad + TILE_H, pad:pad + tile_w] = tile
        for i, line in enumerate(text_lines):
            cv2.putText(canvas, line,
                        (pad, TILE_H + pad + label_h * i + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        return canvas

    # Query crop tile
    query_tile = _make_tile(crop, [label, "QUERY"])

    # Reference tiles
    ref_tiles = []
    for rank, (uid, stem, fpath, ref_img, score) in enumerate(top_entries, 1):
        short_stem = stem if len(stem) <= 14 else stem[:12] + ".."
        ref_tiles.append(_make_tile(ref_img, [
            f"#{rank} {uid}",
            f"{short_stem}",
            f"NCC={score:.3f}",
        ]))

    all_tiles = [query_tile] + ref_tiles
    max_h = max(t.shape[0] for t in all_tiles)
    padded = []
    for t in all_tiles:
        if t.shape[0] < max_h:
            bot = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 30
            t = np.vstack([t, bot])
        padded.append(t)
    return np.hstack(padded)


def main():
    matcher = TemplateMatcher(cell_threshold=0.00)  # threshold=0 so we get all scores
    matcher.load_library(REF_DIR)

    # Load raw template list for per-image scoring
    raw_templates = _load_all_templates(REF_DIR)
    print(f"Loaded {len(raw_templates)} individual template images across all units.")

    with VideoCapture(VIDEO, target_width=1080) as cap:
        cap.detect_game_region()
        first = cap.frame_at(0.0)
        fh, fw = first.shape[:2]
        cal = GridCalibrator.from_defaults(fw, fh)
        print(f"Frame: {fw}x{fh}\n")

        for ts in TIMESTAMPS:
            frame = cap.frame_at(ts)
            if frame is None:
                continue

            for side, row, col in CELLS:
                crop = cal.crop_cell(frame, side, row, col)
                if crop.size == 0:
                    continue

                # Score every individual template image
                scored = []
                for uid, stem, fpath, ref_img in raw_templates:
                    h, w = ref_img.shape[:2]
                    half = cv2.resize(ref_img, (max(1, w // 2), max(1, h // 2)),
                                      interpolation=cv2.INTER_AREA)
                    score = max(_ncc(crop, ref_img), _ncc(crop, half))
                    scored.append((uid, stem, fpath, ref_img, score))

                scored.sort(key=lambda x: x[4], reverse=True)
                top = scored[:TOP_N]

                cell_label = f"t={ts:.0f}s {side} r{row}c{col}"
                print(f"{cell_label}")
                for rank, (uid, stem, fpath, _, score) in enumerate(top, 1):
                    print(f"  #{rank:2d}  {score:.3f}  {uid}/{stem}")
                print()

                # Save comparison image
                strip = make_comparison(crop, top, cell_label)
                fname = f"t{int(ts)}_{side}_r{row}c{col}.png"
                cv2.imwrite(str(OUT_DIR / fname), strip)

    print(f"\nComparison images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()