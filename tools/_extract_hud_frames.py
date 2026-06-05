"""
Extract frames from gameplay video and overlay HUD region boxes
to validate current ocr_reader.py region fractions.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

VIDEO  = Path("data/screenshots/hud_frames/gameplay.mp4")
OUTDIR = Path("data/screenshots/hud_frames")

# Current region fractions from ocr_reader.py
REGIONS = {
    "wave":        (0.22, 0.43, 0.60, 0.49),
    "player_hp":   (0.02, 0.43, 0.28, 0.50),
    "opp_hp":      (0.62, 0.43, 0.97, 0.50),
    "player_mana": (0.00, 0.49, 0.07, 0.72),
}

COLORS = {
    "wave":        (0,   255, 0),    # green
    "player_hp":   (0,   0,   255),  # blue
    "opp_hp":      (255, 0,   0),    # red
    "player_mana": (0,   200, 255),  # yellow
}

def draw_regions(frame, label_prefix=""):
    h, w = frame.shape[:2]
    annotated = frame.copy()
    for name, (l, t, r, b) in REGIONS.items():
        x1, y1 = int(l * w), int(t * h)
        x2, y2 = int(r * w), int(b * h)
        color = COLORS[name]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, name, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    print("ERROR: could not open video"); sys.exit(1)

fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur    = total / fps if fps else 0

print(f"Video: {w}x{h}, {fps:.1f}fps, {total} frames ({dur:.1f}s)")

# Extract frames at 10%, 30%, 50%, 70%, 90% of duration
timestamps = [0.10, 0.30, 0.50, 0.70, 0.90]
saved = []
for pct in timestamps:
    target = int(total * pct)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        continue
    annotated = draw_regions(frame)
    out = OUTDIR / f"frame_{int(pct*100):02d}pct.png"
    cv2.imwrite(str(out), annotated)
    saved.append(out)
    print(f"  Saved {out.name}  (frame {target}/{total})")

# Also save raw (un-annotated) middle frame for clean inspection
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.50))
ok, frame = cap.read()
if ok:
    raw_out = OUTDIR / "frame_50pct_raw.png"
    cv2.imwrite(str(raw_out), frame)
    print(f"  Saved {raw_out.name}  (raw, no annotations)")

cap.release()
print(f"\nDone. {len(saved)} annotated frames in {OUTDIR}")