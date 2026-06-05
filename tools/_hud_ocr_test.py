"""
Quick HUD OCR diagnostic — reads each hud_frames PNG and prints readings.
Also saves cropped region images to data/screenshots/hud_debug/ for visual inspection.
Run from project root:  python tools/_hud_ocr_test.py
"""
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.ocr_reader import OCRReader, _WAVE_REGION, _PLAYER_HP_REGION, _OPP_HP_REGION

FRAMES_DIR = PROJECT_ROOT / "data" / "screenshots" / "hud_frames"
DEBUG_DIR  = PROJECT_ROOT / "data" / "screenshots" / "hud_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

reader = OCRReader()
print(f"Tesseract available: {reader.available}")
if not reader.available:
    print("  (wave readings will be None — HP detection still runs)")
print()

frame_paths = sorted(FRAMES_DIR.glob("*.png"))
if not frame_paths:
    print(f"No PNG files found in {FRAMES_DIR}")
    sys.exit(1)

print(f"{'Frame':<30}  {'Wave':>5}  {'P-HP':>5}  {'O-HP':>5}  {'Size'}")
print("-" * 65)
for path in frame_paths:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"{path.name:<30}  (could not load)")
        continue

    h, w = frame.shape[:2]
    r = reader.read(frame)
    wave = str(r.wave_number) if r.wave_number is not None else "None"
    php  = str(r.player_hp)   if r.player_hp  is not None else "None"
    ohp  = str(r.opponent_hp) if r.opponent_hp is not None else "None"
    print(f"{path.name:<30}  {wave:>5}  {php:>5}  {ohp:>5}  {w}x{h}")

    stem = path.stem
    wave_crop  = reader.crop_region(frame, _WAVE_REGION)
    php_crop   = reader.crop_region(frame, _PLAYER_HP_REGION)
    ohp_crop   = reader.crop_region(frame, _OPP_HP_REGION)
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_wave.png"),    wave_crop)
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_player_hp.png"), php_crop)
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_opp_hp.png"),  ohp_crop)

print()
print(f"Crop images saved to: {DEBUG_DIR}")
print("Done.")