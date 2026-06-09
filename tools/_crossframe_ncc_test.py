"""
Cross-frame NCC test: same board cell at two different timestamps.

If the same unit occupies r0c0 at both frames, the NCC score should be high.
This tells us whether NCC is viable at all for this video quality,
independent of the reference image quality issue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from capture.video_capture import VideoCapture
from capture.grid_calibrator import GridCalibrator

VIDEO = "data/screenshots/hud_frames/gameplay.mp4"

def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    th, tw = b.shape[:2]
    resized = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(resized, b, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])

with VideoCapture(VIDEO) as cap:
    total = cap.frame_count
    frames = {}
    for pct in [0.25, 0.49, 0.74]:
        pos = int(total * pct)
        cap._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, raw = cap._cap.read()
        if ok:
            frames[pct] = cap._process_frame(raw)

pcts = sorted(frames.keys())
fh, fw = frames[pcts[0]].shape[:2]
cal = GridCalibrator.from_defaults(fw, fh)

print("Cross-frame NCC (same cell position, different frames)")
print("A score near 1.0 = same unit, near 0.0 = different unit or noise\n")

# Compare every player cell across all frame pairs
for player in ("player", "opponent"):
    for row in range(5):
        for col in range(3):
            crops = {p: cal.crop_cell(frames[p], player, row, col) for p in pcts}
            # Check all crops are non-trivial
            means = {p: cv2.mean(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))[0]
                     for p, c in crops.items()}
            if any(m < 15 for m in means.values()):
                continue

            scores = []
            for i, p1 in enumerate(pcts):
                for p2 in pcts[i+1:]:
                    s = ncc_score(crops[p1], crops[p2])
                    scores.append(f"{int(p1*100)}%vs{int(p2*100)}%={s:.3f}")

            short = "p" if player == "player" else "o"
            print(f"  {short} r{row}c{col}: {', '.join(scores)}")