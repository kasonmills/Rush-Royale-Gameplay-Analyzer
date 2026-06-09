"""
Test: resize reference template DOWN to deck icon dimensions vs. current approach.
Upscaling blurry 54x76 icons to 100x96 template size loses quality.
Downscaling crisp 270x260 reference to 54x76 icon size should preserve more info.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from capture.video_capture import VideoCapture
from capture.grid_calibrator import GridCalibrator

REFERENCE_DIR = Path("assets/reference")
VIDEO = "data/screenshots/hud_frames/gameplay.mp4"


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC score after resizing b to match a's dimensions."""
    th, tw = a.shape[:2]
    b_resized = cv2.resize(b, (tw, th), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(b_resized, a, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])


def ncc_tmpl_to_icon(icon: np.ndarray, tmpl: np.ndarray) -> float:
    """Current approach: upscale icon to template dimensions."""
    th, tw = tmpl.shape[:2]
    icon_resized = cv2.resize(icon, (tw, th), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(icon_resized, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])


def ncc_icon_to_tmpl(icon: np.ndarray, tmpl: np.ndarray) -> float:
    """New approach: downscale template to icon dimensions."""
    ih, iw = icon.shape[:2]
    tmpl_resized = cv2.resize(tmpl, (iw, ih), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(icon, tmpl_resized, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(result)[1])


# Load reference images
refs: dict[str, np.ndarray] = {}
for unit_dir in sorted(REFERENCE_DIR.iterdir()):
    if not unit_dir.is_dir() or unit_dir.name in ("hero_portraits", "talent_icons"):
        continue
    for img_path in unit_dir.glob("base_rank1.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            refs[unit_dir.name] = img

print(f"Loaded {len(refs)} reference images\n")

# Get deck icons from a gameplay frame
with VideoCapture(VIDEO) as cap:
    total = cap.frame_count
    pos = int(total * 0.49)
    cap._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, raw = cap._cap.read()
    frame = cap._process_frame(raw)
    fh, fw = frame.shape[:2]
    from capture.grid_calibrator import GridCalibrator
    cal = GridCalibrator.from_defaults(fw, fh)
    icons = cal.crop_deck_icons(frame, "player")

print(f"Deck icon size: {icons[0].shape[1]}x{icons[0].shape[0]}px\n")

for slot_idx, icon in enumerate(icons):
    if icon.size == 0:
        continue

    results_old = []
    results_new = []
    for uid, tmpl in refs.items():
        score_old = ncc_tmpl_to_icon(icon, tmpl)   # current: upscale icon
        score_new = ncc_icon_to_tmpl(icon, tmpl)   # new: downscale template
        results_old.append((uid, score_old))
        results_new.append((uid, score_new))

    results_old.sort(key=lambda x: -x[1])
    results_new.sort(key=lambda x: -x[1])

    print(f"Slot {slot_idx}:")
    print(f"  OLD (upscale icon):     top3={[(u, round(s,3)) for u,s in results_old[:3]]}")
    print(f"  NEW (downscale tmpl):   top3={[(u, round(s,3)) for u,s in results_new[:3]]}")
    print()