"""Quick diagnostic: crop player r1c1 at a given timestamp and show NCC scores."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import cv2
from src.capture.grid_calibrator import GridCalibrator
from src.capture.video_capture import VideoCapture
from src.recognition.template_matcher import TemplateMatcher

VIDEO = _ROOT / "data" / "screenshots" / "hud_frames" / "VID_20260624121536.mp4"
REF   = _ROOT / "assets" / "reference"
TIMESTAMPS = [5.0, 15.0, 30.0, 50.0]
CELLS = [("player", 0, 2), ("player", 1, 1), ("player", 2, 2),
         ("opponent", 0, 1), ("opponent", 1, 1)]

matcher = TemplateMatcher(cell_threshold=0.40)
matcher.load_library(REF)

with VideoCapture(VIDEO, target_width=1080) as cap:
    cap.detect_game_region()
    first = cap.frame_at(0.0)
    fh, fw = first.shape[:2]
    cal = GridCalibrator.from_defaults(fw, fh)
    print(f"Frame size: {fw}x{fh}")
    print()

    for ts in TIMESTAMPS:
        frame = cap.frame_at(ts)
        if frame is None:
            print(f"t={ts}s: no frame")
            continue
        print(f"t={ts}s:")
        for side, row, col in CELLS:
            crop = cal.crop_cell(frame, side, row, col)
            result = matcher.match_cell(crop, unit_ids=None)
            if result.is_empty:
                status = "EMPTY"
            elif result.unit_id:
                status = f"{result.unit_id} {result.appearance_state} rank={result.merge_rank} conf={result.confidence:.3f}"
            else:
                status = f"no match (best conf={result.confidence:.3f})"
            print(f"  {side:8} r{row}c{col}: {status}")
            out_path = _ROOT / "data" / "screenshots" / "hud_debug" / f"debug_t{int(ts)}_{side}_r{row}c{col}.png"
            cv2.imwrite(str(out_path), crop)
        print()