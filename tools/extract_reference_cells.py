"""
Extract board cell crops from a gameplay video to build a proper reference library.

The existing reference images are unit CARD screenshots from the collection screen.
In-game, units appear as circular disc sprites that look completely different.
This tool extracts those actual in-game sprites so they can be used as references.

Usage:
    python tools/extract_reference_cells.py data/video/gameplay.mp4

    This samples frames throughout the video, saves cell crops to:
        data/reference_raw/<frame_pct>/player_r<R>c<C>.png
        data/reference_raw/<frame_pct>/opp_r<R>c<C>.png

    Also saves a contact sheet (data/reference_raw/contact_sheet.png) showing
    all crops on one image for visual inspection.

    Once you identify which unit is in each cell, copy the crop to:
        assets/reference/<unit_id>/board_rank<N>.png

    The template matcher will then find it alongside any existing reference images.
"""

import sys
from pathlib import Path

# Allow running from repo root or from tools/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from capture.video_capture import VideoCapture
from capture.grid_calibrator import GridCalibrator


# How many frames to sample (spread evenly across the video).
SAMPLE_COUNT = 12

# Minimum mean pixel value for a cell to be considered non-empty.
_EMPTY_MEAN_THRESHOLD = 20


def _is_empty(cell: np.ndarray) -> bool:
    if cell.size == 0:
        return True
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    return float(mean[0][0]) < _EMPTY_MEAN_THRESHOLD and float(std[0][0]) < 8


def extract_cells(video_path: str):
    out_dir = Path("data/reference_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    with VideoCapture(video_path) as cap:
        total = cap.frame_count
        fps = cap.fps
        print(f"Video: {total} frames @ {fps:.1f} fps  "
              f"({cap.duration_sec:.1f}s)")

        # Sample positions spread across the whole video.
        positions = [int(total * i / SAMPLE_COUNT)
                     for i in range(1, SAMPLE_COUNT + 1)]

        all_crops: list[tuple[str, np.ndarray, np.ndarray]] = []

        for pos in positions:
            cap._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, raw_frame = cap._cap.read()
            if not ok:
                continue

            frame = cap._process_frame(raw_frame)
            fh, fw = frame.shape[:2]
            cal = GridCalibrator.from_defaults(fw, fh)

            pct = int(100 * pos / total)
            frame_dir = out_dir / f"frame_{pct:03d}pct"
            frame_dir.mkdir(exist_ok=True)

            # Save a grid overlay so you can see the cell boundaries.
            grid_img = cal.draw_grid(frame)
            cv2.imwrite(str(frame_dir / "grid_overlay.png"), grid_img)
            print(f"  frame {pct:3d}% -> {frame_dir.name}/  ({fw}x{fh}px)")

            # Save every non-empty cell.
            for player in ("player", "opponent"):
                for row in range(5):
                    for col in range(3):
                        cell = cal.crop_cell(frame, player, row, col)
                        if _is_empty(cell):
                            continue
                        short = "p" if player == "player" else "o"
                        fname = f"{short}_r{row}c{col}.png"
                        path = frame_dir / fname
                        cv2.imwrite(str(path), cell)
                        all_crops.append((str(path), cell, grid_img))

    # Build a contact sheet for easy visual review.
    _save_contact_sheet(out_dir, all_crops)
    print(f"\nDone. {len(all_crops)} non-empty cell crops saved to {out_dir}/")
    print("Open data/reference_raw/contact_sheet.png to review all crops.")
    print("\nNext step:")
    print("  For each crop that shows a known unit, copy it to:")
    print("  assets/reference/<unit_id>/board_rank1.png")
    print("  (Use board_rank1 / board_rank2 etc. to distinguish ranks)")


def _save_contact_sheet(out_dir: Path,
                        crops: list[tuple[str, np.ndarray, np.ndarray]]):
    if not crops:
        print("No crops to make a contact sheet from.")
        return

    # Scale each crop to a fixed display size for the sheet.
    DISPLAY_W, DISPLAY_H = 180, 70
    COLS = 10
    PADDING = 4

    cells_display = []
    for path, cell, _ in crops:
        display = cv2.resize(cell, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
        # Label with the filename.
        label = Path(path).parent.name + "/" + Path(path).name
        cv2.putText(display, label, (2, DISPLAY_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        cells_display.append(display)

    rows = (len(cells_display) + COLS - 1) // COLS
    sheet_w = COLS * (DISPLAY_W + PADDING) + PADDING
    sheet_h = rows * (DISPLAY_H + PADDING) + PADDING
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, img in enumerate(cells_display):
        r = idx // COLS
        c = idx % COLS
        y = PADDING + r * (DISPLAY_H + PADDING)
        x = PADDING + c * (DISPLAY_W + PADDING)
        sheet[y:y + DISPLAY_H, x:x + DISPLAY_W] = img

    out_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"Contact sheet saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/extract_reference_cells.py <video_path>")
        sys.exit(1)
    extract_cells(sys.argv[1])