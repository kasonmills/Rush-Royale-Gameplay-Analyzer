"""
OCR reader for HUD numeric elements: wave number.
Uses pytesseract with per-region pre-processing tuned for the Rush Royale HUD.

Rush Royale PvP HUD layout (portrait orientation, ~1080×2340):
  - Wave number: "Wave N" text in the centre divider strip between the two boards.
  - Player HP:   3 red-heart icon sprites to the LEFT of the centre divider strip.
  - Opponent HP: heart/skull icons to the RIGHT of the centre divider strip.
  - Player mana: crystal icon sprites along the left edge of the player board
                 (not yet implemented — always returns None).

All region positions are expressed as frame fractions so they scale to any
stream resolution. The defaults are tuned for a 360×640 portrait scrcpy stream
and can be overridden by passing a custom HUDLayout.

HP detection uses red-pixel HSV colour counting (connected components), not OCR,
because the HP display is icon-based (heart sprites), not digit text.

Wave OCR pre-processing pipeline:
  1. Convert to grayscale.
  2. Optionally invert (for light text on dark background).
  3. Upscale to ≥ 64 px tall for better OCR accuracy.
  4. Apply Otsu binarisation.
  5. Run pytesseract with --psm 7 (single text line) and a digits-only allowlist.

Usage:
    reader = OCRReader()
    if not reader.available:
        print("Tesseract not found — wave OCR disabled")

    readings = reader.read(frame)
    print(readings.wave_number)  # e.g. 12  (None if Tesseract unavailable)
    print(readings.player_hp)    # e.g. 3   (heart count, always available)
    print(readings.opponent_hp)  # e.g. 2   (heart count, always available)
    print(readings.player_mana)  # None (not yet implemented)
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract as _pytesseract_mod
except ImportError:
    _pytesseract_mod = None  # type: ignore[assignment]


# Minimum pixel height for an OCR crop before it's upscaled.
_MIN_OCR_HEIGHT = 32

# Upscale target height when the crop is below the minimum.
_UPSCALE_TARGET_H = 64

# Tesseract configs tried in order; first valid result wins.
_TESS_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=0123456789",   # single text line
    "--psm 8 -c tessedit_char_whitelist=0123456789",   # single word
    "--psm 13 -c tessedit_char_whitelist=0123456789",  # raw line, no segmentation
]

# Common Tesseract install paths on Windows (checked in order).
_WIN_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]

# Valid range for the wave counter (inclusive).
_WAVE_RANGE = (1, 99)

# Minimum area (px²) for a connected red-pixel cluster to count as a heart.
# Filters single-pixel noise; tuned for a 360×640 source frame.
_MIN_HEART_AREA = 50


# ---------------------------------------------------------------------------
# HUD region definitions — (left_frac, top_frac, right_frac, bottom_frac)
# Calibrated from reference footage (360×640 portrait).
# All regions are in the centre divider strip between the two boards
# (vertically: ~43–50% of frame height).
# ---------------------------------------------------------------------------

# "Wave N" text in the centre divider strip between the two boards.
_WAVE_REGION = (0.28, 0.43, 0.60, 0.49)

# Player castle HP: red heart icons left of centre in the divider strip.
_PLAYER_HP_REGION = (0.02, 0.43, 0.28, 0.50)

# Opponent castle HP: heart/skull icons right of centre — capped at 0.90
# to stay inside the game area (rightmost ~10% is hero portrait sidebar).
_OPP_HP_REGION = (0.58, 0.43, 0.90, 0.50)

# Player mana: crystal icon sprites along the left edge of the player board.
_PLAYER_MANA_REGION = (0.00, 0.49, 0.07, 0.72)


@dataclass
class HUDLayout:
    """
    HUD region fractions (left, top, right, bottom) for each detection target.
    Pass a custom instance to OCRReader to adapt to a different layout.
    """
    wave:        tuple[float, float, float, float] = _WAVE_REGION
    player_hp:   tuple[float, float, float, float] = _PLAYER_HP_REGION
    opponent_hp: tuple[float, float, float, float] = _OPP_HP_REGION
    player_mana: tuple[float, float, float, float] = _PLAYER_MANA_REGION


@dataclass
class HUDReadings:
    """
    Values read from the HUD in a single frame.
    wave_number: None if Tesseract is unavailable or OCR failed.
    player_hp / opponent_hp: heart count (0–3); None only if crop is empty.
    player_mana: always None (not yet implemented).
    """
    wave_number:  Optional[int] = None
    player_hp:    Optional[int] = None
    opponent_hp:  Optional[int] = None
    player_mana:  Optional[int] = None


def _find_tesseract() -> Optional[str]:
    """
    Return the path to the Tesseract executable, or None if not found.

    Checks (in order):
      1. pytesseract's current tesseract_cmd (may already be configured).
      2. Common Windows install locations.
      3. PATH (non-Windows / custom installs).
    """
    if _pytesseract_mod is None:
        return None

    cmd = _pytesseract_mod.pytesseract.tesseract_cmd
    if cmd and cmd != "tesseract" and os.path.isfile(cmd):
        return cmd

    import shutil
    for path in _WIN_TESSERACT_PATHS:
        if os.path.isfile(path):
            return path

    return shutil.which("tesseract")


class OCRReader:
    """
    Reads HUD elements (wave number, HP) from a full frame.

    HP detection works without Tesseract (red-heart colour counting).
    Wave number requires Tesseract — check ``reader.available`` at startup
    and warn the user if it is False.

    Args:
        layout: HUDLayout with region fractions. Defaults to the tuned
                360×640 layout. Override for a different stream resolution.
        invert: If True, inverts the crop before binarisation (use when
                text is dark on a light background). Default False.
    """

    def __init__(self, layout: HUDLayout = HUDLayout(), invert: bool = False):
        self.layout = layout
        self.invert = invert
        self._tesseract_path: Optional[str] = _find_tesseract()
        if self._tesseract_path and _pytesseract_mod is not None:
            _pytesseract_mod.pytesseract.tesseract_cmd = self._tesseract_path

    @property
    def available(self) -> bool:
        """True if Tesseract is installed and reachable (required for wave OCR)."""
        return self._tesseract_path is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, frame: np.ndarray) -> HUDReadings:
        """
        Read all HUD elements from a full frame in one call.

        wave_number requires Tesseract and is None when unavailable.
        HP is always attempted (colour-based; no Tesseract needed).
        """
        return HUDReadings(
            wave_number=self._read_wave(frame) if self.available else None,
            player_hp=self._count_hp(frame, self.layout.player_hp),
            opponent_hp=self._count_hp(frame, self.layout.opponent_hp),
            player_mana=None,
        )

    def read_wave(self, frame: np.ndarray) -> Optional[int]:
        """Read only the wave number from the frame. Requires Tesseract."""
        if not self.available:
            return None
        return self._read_wave(frame)

    def read_hp(self, frame: np.ndarray
                ) -> tuple[Optional[int], Optional[int]]:
        """
        Read both HP values using red-heart colour detection.
        Does not require Tesseract. Returns (player_hp, opponent_hp).
        """
        return (
            self._count_hp(frame, self.layout.player_hp),
            self._count_hp(frame, self.layout.opponent_hp),
        )

    def read_mana(self, frame: np.ndarray) -> Optional[int]:
        """Read player mana. Not yet implemented — always returns None."""
        return None

    def crop_region(self, frame: np.ndarray,
                    region: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop an arbitrary region from a frame using fraction coords.
        Useful for debugging — inspect the raw crop before processing.
        """
        fh, fw = frame.shape[:2]
        l, t, r, b = region
        x1 = max(0, int(fw * l))
        y1 = max(0, int(fh * t))
        x2 = min(fw, int(fw * r))
        y2 = min(fh, int(fh * b))
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_wave(self, frame: np.ndarray) -> Optional[int]:
        raw = self._read_region(frame, self.layout.wave)
        return _validate_range(raw, *_WAVE_RANGE)

    def _count_hp(self, frame: np.ndarray,
                  region: tuple[float, float, float, float]) -> Optional[int]:
        crop = self.crop_region(frame, region)
        return _count_hearts(crop)

    def _read_region(self, frame: np.ndarray,
                     region: tuple[float, float, float, float]) -> Optional[int]:
        """Crop, pre-process, and OCR a single HUD region."""
        crop = self.crop_region(frame, region)
        if crop.size == 0:
            return None
        preprocessed = _preprocess(crop, self.invert)
        return _run_ocr(preprocessed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_range(value: Optional[int], lo: int, hi: int) -> Optional[int]:
    """Return value if lo ≤ value ≤ hi, else None."""
    if value is None:
        return None
    return value if lo <= value <= hi else None


def _count_hearts(crop: np.ndarray) -> Optional[int]:
    """
    Count red heart icons in a HUD crop using HSV colour segmentation.

    Red hue wraps in HSV (0–15 and 165–180); both ranges are merged.
    Clusters smaller than _MIN_HEART_AREA px² are treated as noise.
    Returns an int in [0, 3], or None if the crop is empty.
    """
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_lo = cv2.inRange(hsv, np.array([0,   100, 80]), np.array([15,  255, 255]))
    mask_hi = cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask_lo, mask_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hearts = sum(
        1 for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= _MIN_HEART_AREA
    )
    return min(hearts, 3)


def _preprocess(crop: np.ndarray, invert: bool) -> np.ndarray:
    """
    Prepare a HUD crop for pytesseract digit recognition.

    Steps:
      1. Grayscale.
      2. Optional invert (dark text on light BG).
      3. Upscale if crop is shorter than _MIN_OCR_HEIGHT.
      4. Otsu binarisation.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    h = gray.shape[0]
    if h < _MIN_OCR_HEIGHT:
        scale = _UPSCALE_TARGET_H / h
        new_w = max(1, int(gray.shape[1] * scale))
        gray = cv2.resize(gray, (new_w, _UPSCALE_TARGET_H),
                          interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _run_ocr(binary: np.ndarray) -> Optional[int]:
    """
    Run pytesseract on a pre-processed binary image and return an integer,
    or None if the result is empty or non-numeric.

    Tries multiple PSM modes in order; returns the first valid digit string.
    """
    for cfg in _TESS_CONFIGS:
        text = _pytesseract_mod.image_to_string(binary, config=cfg).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return int(digits)
    return None