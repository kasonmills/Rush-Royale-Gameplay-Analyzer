"""Quick test: NCC scores for deck icons against the reference library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from capture.video_capture import VideoCapture
from capture.grid_calibrator import GridCalibrator
from recognition.template_matcher import (
    TemplateMatcher, _max_ncc_score, _is_empty_cell
)

VIDEO = "data/screenshots/hud_frames/gameplay.mp4"
SAMPLE_PCTS = [0.25, 0.49, 0.74]

matcher = TemplateMatcher(deck_icon_threshold=0.0)
matcher.load_library("assets/reference")
print(f"Library: {len(matcher._index)} units\n")

with VideoCapture(VIDEO) as cap:
    total = cap.frame_count
    for pct in SAMPLE_PCTS:
        pos = int(total * pct)
        cap._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, raw = cap._cap.read()
        if not ok:
            continue
        frame = cap._process_frame(raw)
        fh, fw = frame.shape[:2]
        cal = GridCalibrator.from_defaults(fw, fh)

        print(f"=== Frame {int(pct*100)}% ===")
        icons = cal.crop_deck_icons(frame, "player")
        print(f"  Icon sizes: {[f'{i.shape[1]}x{i.shape[0]}' for i in icons]}")

        for i, icon in enumerate(icons):
            if _is_empty_cell(icon):
                print(f"  slot {i}: empty")
                continue
            scores = []
            for uid, entries in matcher._index.items():
                best = max(_max_ncc_score(icon, e) for e in entries)
                scores.append((uid, round(best, 3)))
            scores.sort(key=lambda x: -x[1])
            top5 = scores[:5]
            print(f"  slot {i} ({icon.shape[1]}x{icon.shape[0]}): {top5}")
        print()