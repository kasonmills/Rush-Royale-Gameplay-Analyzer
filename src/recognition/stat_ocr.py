"""
OCR reader for per-unit stat numbers displayed on board tiles in Rush Royale.

Many Rush Royale units display a numeric counter on their tile that tracks
a mechanic (soul count, charge count, leaf count, etc.). This module reads
those numbers by cropping the bottom portion of the cell and running
pytesseract.

Not all units have a stat number; the registry (loaded from Stat Numbers.csv
or unit_meta.db) tracks which units do and what the valid value range is.

Units with position='bottom_panel' show their rank-sum in the deck strip area.
Reading those requires a full-frame crop — currently not implemented; read()
returns None for bottom_panel entries.

Usage:
    ocr = StatOCR()
    ocr.load_from_csv("data/Stat Numbers.csv")
    if not ocr.available:
        print("Tesseract not found — stat OCR disabled")

    value = ocr.read(cell_crop, "alchemist")
    value = ocr.read(cell_crop, "crystalmancer", talent_branch="R", talent_tier=1)
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract as _pytesseract_mod
except ImportError:
    _pytesseract_mod = None  # type: ignore[assignment]


_TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

_WIN_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]

# Minimum pixel height before upscaling for OCR.
_MIN_OCR_HEIGHT = 20
_UPSCALE_TARGET_H = 48

# Fraction of cell height from the bottom that contains the stat counter.
# Rush Royale renders tile stat numbers in the lower portion of the unit tile.
_STAT_REGION_BOTTOM_FRAC = 0.35


@dataclass
class StatEntry:
    """
    One row from the Stat Numbers registry.
    Describes a numeric counter displayed on a specific unit's tile.
    """
    unit_id: str
    position: str                    # 'tile' | 'bottom_panel'
    meaning: str                     # human-readable description
    talent_branch: Optional[str]     # None = branch-independent
    talent_tier: Optional[int]       # None = tier-independent
    max_value: int                   # inclusive upper bound for range validation


def _find_tesseract() -> Optional[str]:
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


def _crop_tile_stat_region(cell_crop: np.ndarray) -> np.ndarray:
    """
    Return the bottom _STAT_REGION_BOTTOM_FRAC of the cell — the area where
    Rush Royale renders tile stat counters.
    """
    h = cell_crop.shape[0]
    y_start = max(0, int(h * (1.0 - _STAT_REGION_BOTTOM_FRAC)))
    return cell_crop[y_start:, :]


def _validate_stat(value: Optional[int], max_value: int) -> Optional[int]:
    """Return value if 0 ≤ value ≤ max_value, else None."""
    if value is None:
        return None
    return value if 0 <= value <= max_value else None


def _run_stat_ocr(region: np.ndarray) -> Optional[int]:
    """
    Preprocess a stat region and run pytesseract.

    Tries inverted first (black text on white — Tesseract-friendly for the
    white-on-dark numbers common in Rush Royale), then non-inverted as a
    fallback for any light-background variants.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    for do_invert in (True, False):
        g = cv2.bitwise_not(gray) if do_invert else gray

        h = g.shape[0]
        if h < _MIN_OCR_HEIGHT:
            scale = _UPSCALE_TARGET_H / max(h, 1)
            new_w = max(1, int(g.shape[1] * scale))
            g = cv2.resize(g, (new_w, _UPSCALE_TARGET_H),
                           interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(g, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = _pytesseract_mod.image_to_string(binary,
                                                config=_TESS_CONFIG).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return int(digits)

    return None


class StatOCR:
    """
    Reads per-unit stat counters from board cell crops.

    Load the registry first with load_from_csv() or load_from_db(), then
    call read() for each occupied cell.  Units not in the registry, units
    whose stat is at a 'bottom_panel' position, and cells where OCR fails
    all return None.

    Args:
        tesseract_path: Override the Tesseract executable path. If None,
                        common Windows install paths and PATH are searched.
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        self._registry: dict[str, list[StatEntry]] = {}
        self._tesseract_path: Optional[str] = tesseract_path or _find_tesseract()
        if self._tesseract_path and _pytesseract_mod is not None:
            _pytesseract_mod.pytesseract.tesseract_cmd = self._tesseract_path

    @property
    def available(self) -> bool:
        """True if Tesseract is installed and reachable."""
        return self._tesseract_path is not None

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def load_from_csv(self, path: str | Path) -> None:
        """
        Populate the registry from Stat Numbers.csv.

        Expected columns (by name):
          Unit ID, Talent Branch, Talent Tier, Number Position,
          What does this number mean?, Max Observed Value
        """
        self._registry.clear()
        with open(Path(path), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                unit_id = row.get("Unit ID", "").strip()
                if not unit_id:
                    continue

                position = row.get("Number Position", "").strip()
                meaning  = row.get("What does this number mean?", "").strip()
                branch   = row.get("Talent Branch", "").strip() or None

                tier_raw = row.get("Talent Tier", "").strip()
                tier: Optional[int] = int(tier_raw) if tier_raw.isdigit() else None

                max_raw = row.get("Max Observed Value", "0").strip()
                try:
                    max_value = int(float(max_raw)) if max_raw else 0
                except ValueError:
                    max_value = 0

                entry = StatEntry(
                    unit_id=unit_id,
                    position=position,
                    meaning=meaning,
                    talent_branch=branch,
                    talent_tier=tier,
                    max_value=max_value,
                )
                self._registry.setdefault(unit_id, []).append(entry)

    def load_from_db(self, conn) -> None:
        """
        Populate the registry from the stat_numbers table in unit_meta.db.
        Clears any previously loaded entries.
        """
        import json

        self._registry.clear()
        rows = conn.execute(
            """
            SELECT unit_id, talent_branch, talent_tier, position,
                   meaning, scaling_formula
            FROM stat_numbers
            """
        ).fetchall()

        for row in rows:
            sf = row["scaling_formula"]
            try:
                data = json.loads(sf) if sf else {}
                segs = data.get("segments", [])
                max_value = int(segs[-1]["to"]) if segs else 999
            except (json.JSONDecodeError, KeyError, TypeError):
                max_value = 999

            tier = row["talent_tier"]
            entry = StatEntry(
                unit_id=row["unit_id"],
                position=row["position"],
                meaning=row["meaning"] or "",
                talent_branch=row["talent_branch"],
                talent_tier=int(tier) if tier is not None else None,
                max_value=max_value,
            )
            self._registry.setdefault(entry.unit_id, []).append(entry)

    def known_unit_ids(self) -> set[str]:
        """Return the set of unit_ids that have at least one stat entry."""
        return set(self._registry.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self,
             cell_crop: np.ndarray,
             unit_id: str,
             talent_branch: Optional[str] = None,
             talent_tier: Optional[int] = None) -> Optional[int]:
        """
        Read the stat counter from a board cell crop.

        Args:
            cell_crop:     BGR image of a single board cell.
            unit_id:       The unit occupying the cell (matches unit_meta.db).
            talent_branch: Active talent branch ('L', 'R', 'Fixed', or None).
            talent_tier:   Highest active talent tier (1-4, or None if unknown).

        Returns:
            The integer stat value, or None if:
              - Tesseract is unavailable
              - The unit has no tile-position stat in the registry
              - OCR produced no readable digits
              - The value is outside [0, max_value]
        """
        if not self.available or cell_crop.size == 0:
            return None

        entry = self._find_entry(unit_id, talent_branch, talent_tier)
        if entry is None:
            return None

        region = _crop_tile_stat_region(cell_crop)
        if region.size == 0:
            return None

        raw = _run_stat_ocr(region)
        return _validate_stat(raw, entry.max_value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_entry(self,
                    unit_id: str,
                    branch: Optional[str],
                    tier: Optional[int]) -> Optional[StatEntry]:
        """
        Return the best-matching tile-position StatEntry for the given context.

        Only tile-position entries are returned; bottom_panel entries are
        excluded because they require a full-frame crop, not a cell crop.

        Matching priority (most specific first):
          1. Exact branch AND tier
          2. Exact branch, no tier constraint in entry
          3. Branch-independent entry (talent_branch is None)
          4. First tile entry in registry (last resort)
        Returns None if the unit has no tile-position entries.
        """
        all_entries = self._registry.get(unit_id)
        if not all_entries:
            return None

        tile = [e for e in all_entries if e.position == "tile"]
        if not tile:
            return None

        # 1. Exact branch + tier
        if branch is not None and tier is not None:
            for e in tile:
                if e.talent_branch == branch and e.talent_tier == tier:
                    return e

        # 2. Exact branch, any tier
        if branch is not None:
            for e in tile:
                if e.talent_branch == branch and e.talent_tier is None:
                    return e

        # 3. Branch-independent
        for e in tile:
            if e.talent_branch is None and e.talent_tier is None:
                return e

        # 4. First tile entry
        return tile[0]