"""
OCR reader for HUD numeric elements: castle HP, wave number, and mana.
Uses pytesseract with per-region pre-processing tuned for the Rush Royale HUD.

Rush Royale PvP HUD layout (portrait orientation, ~1080×2340):
  - Wave number: displayed between the two boards, centre of the frame.
  - Player HP:   castle hit-points shown near the player HP bar (lower middle).
  - Opponent HP: castle hit-points shown near the opponent HP bar (upper middle).
  - Player mana: mana value shown near the player board (lower area).

All region positions are expressed as frame fractions so they scale to any
stream resolution. The defaults are tuned for a 1080×2340 scrcpy stream and
can be overridden by passing a custom HUDLayout.

Pre-processing pipeline per region:
  1. Convert to grayscale.
  2. Optionally invert (for light text on dark background).
  3. Upscale to ≥ 64 px tall for better OCR accuracy.
  4. Apply Otsu binarisation.
  5. Run pytesseract with --psm 7 (single text line) and a digits-only allowlist.

Usage:
    reader = OCRReader()
    if not reader.available:
        print("Tesseract not found — install from https://github.com/UB-Mannheim/tesseract/wiki")

    readings = reader.read(frame)
    print(readings.wave_number)  # e.g. 12
    print(readings.player_hp)    # e.g. 3
    print(readings.opponent_hp)  # e.g. 2
    print(readings.player_mana)  # e.g. 4
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
# tesseract accuracy degrades significantly on crops shorter than this.
_MIN_OCR_HEIGHT = 32

# Upscale target height when the crop is below the minimum.
_UPSCALE_TARGET_H = 64

# Tesseract config shared by all digit reads.
_TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

# Common Tesseract install paths on Windows (checked in order).
_WIN_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]

# Valid ranges for each HUD field (inclusive).  Values outside these
# bounds are treated as OCR noise and discarded (returned as None).
_WAVE_RANGE  = (1, 99)
_HP_RANGE    = (0, 3)    # PvP: each player has 3 lives
_MANA_RANGE  = (0, 9)


# ---------------------------------------------------------------------------
# HUD region definitions — (left_frac, top_frac, right_frac, bottom_frac)
# Tuned for a 1080×2340 portrait scrcpy stream.
# ---------------------------------------------------------------------------

# Wave number counter displayed in the centre between the two boards.
_WAVE_REGION     = (0.38, 0.47, 0.62, 0.53)

# Player castle HP: the numeric value near the player HP bar, lower centre.
_PLAYER_HP_REGION = (0.30, 0.92, 0.70, 0.99)

# Opponent castle HP: upper centre, symmetric to player HP.
_OPP_HP_REGION   = (0.30, 0.01, 0.70, 0.08)

# Player mana: displayed near the mana counter below the player board.
_PLAYER_MANA_REGION = (0.42, 0.88, 0.58, 0.93)


@dataclass
class HUDLayout:
    """
    HUD region fractions (left, top, right, bottom) for each OCR target.
    Pass a custom instance to OCRReader to adapt to a different layout.
    """
    wave:        tuple[float, float, float, float] = _WAVE_REGION
    player_hp:   tuple[float, float, float, float] = _PLAYER_HP_REGION
    opponent_hp: tuple[float, float, float, float] = _OPP_HP_REGION
    player_mana: tuple[float, float, float, float] = _PLAYER_MANA_REGION


@dataclass
class HUDReadings:
    """
    Parsed numeric values read from the HUD in a single frame.
    Any field is None if OCR could not produce a valid number for that element.
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
    # Windows common paths
    for path in _WIN_TESSERACT_PATHS:
        if os.path.isfile(path):
            return path

    # Fall back to PATH lookup
    return shutil.which("tesseract")


class OCRReader:
    """
    Reads numeric HUD elements (wave, HP, mana) from a full frame using
    pytesseract.

    If Tesseract is not installed, all reads return None instead of raising —
    check ``reader.available`` at startup and warn the user if it is False.

    Args:
        layout: HUDLayout with region fractions. Defaults to the tuned
                1080×2340 layout. Override for a different stream size or
                UI variant.
        invert: If True, inverts the crop before binarisation (use when
                text is dark on a light background). Default False
                (light text on dark HUD).
    """

    def __init__(self, layout: HUDLayout = HUDLayout(), invert: bool = False):
        self.layout = layout
        self.invert = invert
        self._tesseract_path: Optional[str] = _find_tesseract()
        if self._tesseract_path and _pytesseract_mod is not None:
            _pytesseract_mod.pytesseract.tesseract_cmd = self._tesseract_path

    @property
    def available(self) -> bool:
        """True if Tesseract is installed and reachable."""
        return self._tesseract_path is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, frame: np.ndarray) -> HUDReadings:
        """
        Read all HUD numeric elements from a full frame in one call.

        Returns HUDReadings with each field set to the parsed integer value,
        or None if the region could not be read cleanly or Tesseract is
        unavailable.
        """
        if not self.available:
            return HUDReadings()
        return HUDReadings(
            wave_number=self._read_wave(frame),
            player_hp=self._read_hp_value(frame, self.layout.player_hp),
            opponent_hp=self._read_hp_value(frame, self.layout.opponent_hp),
            player_mana=self._read_mana(frame),
        )

    def read_wave(self, frame: np.ndarray) -> Optional[int]:
        """Read only the wave number from the frame."""
        if not self.available:
            return None
        return self._read_wave(frame)

    def read_hp(self, frame: np.ndarray
                ) -> tuple[Optional[int], Optional[int]]:
        """
        Read both HP values from the frame.
        Returns (player_hp, opponent_hp).
        """
        if not self.available:
            return (None, None)
        return (
            self._read_hp_value(frame, self.layout.player_hp),
            self._read_hp_value(frame, self.layout.opponent_hp),
        )

    def read_mana(self, frame: np.ndarray) -> Optional[int]:
        """Read only the player mana from the frame."""
        if not self.available:
            return None
        return self._read_mana(frame)

    def crop_region(self, frame: np.ndarray,
                    region: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop an arbitrary region from a frame using fraction coords.
        Useful for debugging — inspect the raw crop before OCR.
        """
        fh, fw = frame.shape[:2]
        l, t, r, b = region
        x1 = max(0, int(fw * l))
        y1 = max(0, int(fh * t))
        x2 = min(fw, int(fw * r))
        y2 = min(fh, int(fh * b))
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Internal — typed reads with range validation
    # ------------------------------------------------------------------

    def _read_wave(self, frame: np.ndarray) -> Optional[int]:
        raw = self._read_region(frame, self.layout.wave)
        return _validate_range(raw, *_WAVE_RANGE)

    def _read_hp_value(self, frame: np.ndarray,
                       region: tuple[float, float, float, float]) -> Optional[int]:
        raw = self._read_region(frame, region)
        return _validate_range(raw, *_HP_RANGE)

    def _read_mana(self, frame: np.ndarray) -> Optional[int]:
        raw = self._read_region(frame, self.layout.player_mana)
        return _validate_range(raw, *_MANA_RANGE)

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
    """
    text = _pytesseract_mod.image_to_string(binary, config=_TESS_CONFIG).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)
