"""
Tests for StatOCR.

Covers:
  - _crop_tile_stat_region(): correct bottom fraction, width preserved
  - _validate_stat(): in-range pass-through, boundary values, out-of-range → None
  - StatOCR.load_from_csv(): populates registry, branch/tier parsing
  - StatOCR._find_entry(): priority cascade, tile-only filtering
  - StatOCR.available: reflects Tesseract path
  - StatOCR.read(): None when unavailable, unknown unit, bottom_panel unit;
                    correct value + range validation when pytesseract mocked

pytesseract is always mocked so tests run without a Tesseract installation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.recognition.stat_ocr import (
    StatEntry,
    StatOCR,
    _crop_tile_stat_region,
    _validate_stat,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAT_CSV = PROJECT_ROOT / "data" / "Stat Numbers.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_cell(h: int = 100, w: int = 100,
                color: tuple = (128, 128, 128)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _make_ocr_no_tesseract() -> StatOCR:
    with patch("src.recognition.stat_ocr._find_tesseract", return_value=None):
        ocr = StatOCR()
    return ocr


def _make_ocr_with_tesseract(path: str = r"C:\fake\tesseract.exe") -> StatOCR:
    with patch("src.recognition.stat_ocr._find_tesseract", return_value=path):
        ocr = StatOCR()
    ocr._tesseract_path = path
    return ocr


def _registry_with_entries() -> dict:
    return {
        "alchemist": [
            StatEntry("alchemist", "tile",         "elixirs",  None, None, 200),
            StatEntry("alchemist", "bottom_panel",  "rank sum", None, None,  49),
        ],
        "crystalmancer": [
            StatEntry("crystalmancer", "tile", "charges", "R", 1, 2000),
        ],
        "demon_hunter": [
            StatEntry("demon_hunter", "tile",         "L marks",  "L", 1, 200),
            StatEntry("demon_hunter", "tile",         "R marks",  "R", 1, 200),
            StatEntry("demon_hunter", "bottom_panel", "rank sum", None, None, 49),
        ],
        "banshee": [
            StatEntry("banshee", "bottom_panel", "rank sum", None, None, 42),
        ],
    }


# ---------------------------------------------------------------------------
# _crop_tile_stat_region
# ---------------------------------------------------------------------------

class TestCropTileStatRegion:

    def test_returns_bottom_35_percent_of_height(self):
        cell = _solid_cell(100, 80)
        region = _crop_tile_stat_region(cell)
        assert region.shape[0] == 35

    def test_preserves_full_width(self):
        cell = _solid_cell(100, 80)
        region = _crop_tile_stat_region(cell)
        assert region.shape[1] == 80

    def test_small_cell_returns_nonzero_region(self):
        cell = _solid_cell(10, 10)
        region = _crop_tile_stat_region(cell)
        assert region.size > 0

    def test_region_is_from_bottom(self):
        """Bottom pixels of cell should match bottom pixels of region."""
        cell = np.zeros((100, 10, 3), dtype=np.uint8)
        cell[90:, :] = 255  # white band at very bottom
        region = _crop_tile_stat_region(cell)
        assert region[-1, 0, 0] == 255


# ---------------------------------------------------------------------------
# _validate_stat
# ---------------------------------------------------------------------------

class TestValidateStat:

    def test_zero_is_valid(self):
        assert _validate_stat(0, 100) == 0

    def test_max_value_is_valid(self):
        assert _validate_stat(100, 100) == 100

    def test_midrange_value_returned(self):
        assert _validate_stat(42, 100) == 42

    def test_above_max_returns_none(self):
        assert _validate_stat(101, 100) is None

    def test_negative_returns_none(self):
        assert _validate_stat(-1, 100) is None

    def test_none_input_returns_none(self):
        assert _validate_stat(None, 100) is None


# ---------------------------------------------------------------------------
# StatOCR.load_from_csv
# ---------------------------------------------------------------------------

class TestLoadFromCSV:

    def _load(self) -> StatOCR:
        if not STAT_CSV.exists():
            pytest.skip("data/Stat Numbers.csv not found")
        ocr = StatOCR.__new__(StatOCR)
        ocr._registry = {}
        ocr.load_from_csv(STAT_CSV)
        return ocr

    def test_alchemist_loaded(self):
        ocr = self._load()
        assert "alchemist" in ocr.known_unit_ids()

    def test_tile_and_bottom_panel_both_loaded_for_alchemist(self):
        ocr = self._load()
        positions = {e.position for e in ocr._registry["alchemist"]}
        assert "tile" in positions
        assert "bottom_panel" in positions

    def test_branch_specific_entry_parsed(self):
        ocr = self._load()
        entries = ocr._registry.get("crystalmancer", [])
        assert any(e.talent_branch == "R" and e.talent_tier == 1 for e in entries)

    def test_branch_independent_entry_has_none_branch(self):
        ocr = self._load()
        entries = ocr._registry.get("alchemist", [])
        tile_entries = [e for e in entries if e.position == "tile"]
        assert any(e.talent_branch is None for e in tile_entries)

    def test_max_value_parsed(self):
        ocr = self._load()
        entries = ocr._registry.get("alchemist", [])
        tile = next(e for e in entries if e.position == "tile")
        assert tile.max_value > 0

    def test_known_unit_ids_nonempty(self):
        ocr = self._load()
        assert len(ocr.known_unit_ids()) > 0


# ---------------------------------------------------------------------------
# StatOCR._find_entry
# ---------------------------------------------------------------------------

class TestFindEntry:

    def _ocr_with_registry(self) -> StatOCR:
        ocr = StatOCR.__new__(StatOCR)
        ocr._registry = _registry_with_entries()
        return ocr

    def test_unknown_unit_returns_none(self):
        ocr = self._ocr_with_registry()
        assert ocr._find_entry("phantom", None, None) is None

    def test_branch_independent_tile_entry_found(self):
        ocr = self._ocr_with_registry()
        entry = ocr._find_entry("alchemist", None, None)
        assert entry is not None
        assert entry.position == "tile"
        assert entry.talent_branch is None

    def test_exact_branch_and_tier_matched(self):
        ocr = self._ocr_with_registry()
        entry = ocr._find_entry("crystalmancer", "R", 1)
        assert entry is not None
        assert entry.talent_branch == "R"
        assert entry.talent_tier == 1

    def test_correct_branch_selected_for_demon_hunter(self):
        ocr = self._ocr_with_registry()
        entry = ocr._find_entry("demon_hunter", "R", 1)
        assert entry is not None
        assert entry.talent_branch == "R"

    def test_wrong_branch_falls_back_to_branch_independent(self):
        """alchemist has a branch-independent tile entry; L branch should find it."""
        ocr = self._ocr_with_registry()
        entry = ocr._find_entry("alchemist", "L", 1)
        assert entry is not None
        assert entry.talent_branch is None

    def test_bottom_panel_only_unit_returns_none(self):
        """banshee has only a bottom_panel entry — tile-only filtering returns None."""
        ocr = self._ocr_with_registry()
        assert ocr._find_entry("banshee", None, None) is None


# ---------------------------------------------------------------------------
# StatOCR.available
# ---------------------------------------------------------------------------

class TestAvailable:

    def test_unavailable_without_tesseract(self):
        ocr = _make_ocr_no_tesseract()
        assert ocr.available is False

    def test_available_with_tesseract(self):
        ocr = _make_ocr_with_tesseract()
        assert ocr.available is True


# ---------------------------------------------------------------------------
# StatOCR.read
# ---------------------------------------------------------------------------

class TestStatOCRRead:

    def _make_loaded_ocr(self, mock_text: str) -> StatOCR:
        ocr = _make_ocr_with_tesseract()
        ocr._registry = _registry_with_entries()
        mock_pts = MagicMock()
        mock_pts.image_to_string.return_value = mock_text
        ocr._mock_pts = mock_pts
        return ocr

    def _run_read(self, ocr: StatOCR, unit_id: str,
                  branch: str = None, tier: int = None) -> object:
        cell = _solid_cell(100, 100)
        with patch("src.recognition.stat_ocr._pytesseract_mod", ocr._mock_pts):
            return ocr.read(cell, unit_id,
                            talent_branch=branch, talent_tier=tier)

    def test_returns_none_without_tesseract(self):
        ocr = _make_ocr_no_tesseract()
        ocr._registry = _registry_with_entries()
        assert ocr.read(_solid_cell(), "alchemist") is None

    def test_returns_none_for_empty_crop(self):
        ocr = _make_ocr_with_tesseract()
        ocr._registry = _registry_with_entries()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert ocr.read(empty, "alchemist") is None

    def test_returns_none_for_unknown_unit(self):
        ocr = self._make_loaded_ocr("50")
        assert self._run_read(ocr, "phantom") is None

    def test_returns_none_for_bottom_panel_unit(self):
        """banshee has only a bottom_panel entry."""
        ocr = self._make_loaded_ocr("21")
        assert self._run_read(ocr, "banshee") is None

    def test_valid_value_returned(self):
        ocr = self._make_loaded_ocr("75")
        assert self._run_read(ocr, "alchemist") == 75

    def test_zero_is_valid(self):
        ocr = self._make_loaded_ocr("0")
        assert self._run_read(ocr, "alchemist") == 0

    def test_value_above_max_returns_none(self):
        """Alchemist max is 200; OCR returning 999 should be rejected."""
        ocr = self._make_loaded_ocr("999")
        assert self._run_read(ocr, "alchemist") is None

    def test_branch_matched_for_crystalmancer(self):
        ocr = self._make_loaded_ocr("350")
        result = self._run_read(ocr, "crystalmancer", branch="R", tier=1)
        assert result == 350

    def test_noisy_ocr_digits_extracted(self):
        """'l23' (OCR noise) should parse as 23."""
        ocr = self._make_loaded_ocr("l23")
        assert self._run_read(ocr, "alchemist") == 23

    def test_empty_ocr_returns_none(self):
        ocr = self._make_loaded_ocr("")
        assert self._run_read(ocr, "alchemist") is None