"""
Tests for AnimationDetector.

Covers:
  - _mean_saturation(): correct HSV-S measurement
  - _to_snake(): animation_id derivation
  - AnimationDetector._is_anomalous(): absolute and delta modes
  - AnimationDetector.detect(): empty crop, no anomaly, anomaly with/without registry
  - AnimationDetector.load_from_csv(): registry population, modifier parsing
  - AnimationDetector.load_from_db(): registry population from schema
  - AnimationDetector.known_unit_ids(): reflects loaded registry
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.recognition.animation_detector import (
    AnimationDetector,
    AnimationEntry,
    _mean_saturation,
    _to_snake,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIM_CSV = PROJECT_ROOT / "data" / "Animations.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_bgr(h: int = 100, w: int = 100,
               b: int = 0, g: int = 0, r: int = 0) -> np.ndarray:
    return np.full((h, w, 3), [b, g, r], dtype=np.uint8)


def _gray_cell(sat: int = 0, h: int = 100, w: int = 100) -> np.ndarray:
    """HSV: fixed hue 0, fixed S=sat, fixed V=200. Convert to BGR for testing."""
    import cv2
    hsv = np.full((h, w, 3), [0, sat, 200], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_detector(sat_threshold: int = 80,
                   delta_threshold: int = 40) -> AnimationDetector:
    return AnimationDetector(sat_threshold=sat_threshold,
                             delta_threshold=delta_threshold)


def _registry_detector() -> AnimationDetector:
    det = _make_detector()
    det._registry = {
        "engineer": [
            AnimationEntry("engineer", "eng_chain_glow", "Engineer Chain Glow",
                           "Intrinsic Buff", True, 1.0),
        ],
    }
    return det


# ---------------------------------------------------------------------------
# _mean_saturation
# ---------------------------------------------------------------------------

class TestMeanSaturation:

    def test_gray_has_zero_saturation(self):
        gray = _gray_cell(sat=0)
        assert _mean_saturation(gray) == pytest.approx(0.0, abs=2)

    def test_vivid_has_high_saturation(self):
        vivid = _gray_cell(sat=200)
        assert _mean_saturation(vivid) >= 190

    def test_pure_red_has_near_max_saturation(self):
        red = _solid_bgr(r=255, g=0, b=0)
        assert _mean_saturation(red) >= 200


# ---------------------------------------------------------------------------
# _to_snake
# ---------------------------------------------------------------------------

class TestToSnake:

    def test_simple_name(self):
        assert _to_snake("Engineer Chain Glow") == "engineer_chain_glow"

    def test_leading_trailing_spaces_stripped(self):
        assert _to_snake("  fire aura  ") == "fire_aura"

    def test_punctuation_replaced(self):
        assert _to_snake("Buff (active)") == "buff_active"

    def test_already_snake(self):
        assert _to_snake("eng_chain_glow") == "eng_chain_glow"


# ---------------------------------------------------------------------------
# AnimationDetector._is_anomalous
# ---------------------------------------------------------------------------

class TestIsAnomalous:

    def test_absolute_mode_below_threshold_is_not_anomalous(self):
        det = _make_detector(sat_threshold=80)
        low_sat = _gray_cell(sat=40)
        assert det._is_anomalous(low_sat, None) is False

    def test_absolute_mode_above_threshold_is_anomalous(self):
        det = _make_detector(sat_threshold=80)
        high_sat = _gray_cell(sat=150)
        assert det._is_anomalous(high_sat, None) is True

    def test_absolute_mode_at_threshold_is_anomalous(self):
        det = _make_detector(sat_threshold=80)
        at_threshold = _gray_cell(sat=80)
        assert det._is_anomalous(at_threshold, None) is True

    def test_delta_mode_below_delta_is_not_anomalous(self):
        det = _make_detector(delta_threshold=40)
        reference = _gray_cell(sat=100)
        cell = _gray_cell(sat=120)  # delta = 20, below 40
        assert det._is_anomalous(cell, reference) is False

    def test_delta_mode_above_delta_is_anomalous(self):
        det = _make_detector(delta_threshold=40)
        reference = _gray_cell(sat=50)
        cell = _gray_cell(sat=120)  # delta ≈ 70, above 40
        assert det._is_anomalous(cell, reference) is True

    def test_delta_mode_tiny_reference_falls_back_to_absolute(self):
        det = _make_detector(sat_threshold=80, delta_threshold=40)
        tiny_ref = np.zeros((2, 2, 3), dtype=np.uint8)  # size < _MIN_CELL_AREA
        high_sat = _gray_cell(sat=150)
        # Falls back to absolute since reference is too small
        assert det._is_anomalous(high_sat, tiny_ref) is True


# ---------------------------------------------------------------------------
# AnimationDetector.detect
# ---------------------------------------------------------------------------

class TestDetect:

    def test_empty_crop_returns_empty_list(self):
        det = _make_detector()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert det.detect(empty, "engineer") == []

    def test_tiny_crop_returns_empty_list(self):
        det = _make_detector()
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)  # 25 px² < _MIN_CELL_AREA
        assert det.detect(tiny, "engineer") == []

    def test_no_anomaly_returns_empty_list(self):
        det = _registry_detector()
        low_sat = _gray_cell(sat=20)
        assert det.detect(low_sat, "engineer") == []

    def test_anomaly_with_registry_returns_animation_ids(self):
        det = _registry_detector()
        high_sat = _gray_cell(sat=200)
        result = det.detect(high_sat, "engineer")
        assert result == ["eng_chain_glow"]

    def test_anomaly_without_registry_returns_generic_sentinel(self):
        det = _make_detector()
        high_sat = _gray_cell(sat=200)
        result = det.detect(high_sat, "unknown_unit")
        assert result == ["buff_active"]

    def test_anomaly_unit_not_in_registry_returns_sentinel(self):
        det = _registry_detector()
        high_sat = _gray_cell(sat=200)
        result = det.detect(high_sat, "archer")  # archer not in registry
        assert result == ["buff_active"]

    def test_reference_mode_no_delta_returns_empty(self):
        det = _registry_detector()
        # Both cell and reference have same saturation → delta = 0
        reference = _gray_cell(sat=150)
        cell = _gray_cell(sat=155)  # delta well below 40
        assert det.detect(cell, "engineer", reference_crop=reference) == []

    def test_reference_mode_large_delta_detects(self):
        det = _registry_detector()
        reference = _gray_cell(sat=30)
        cell = _gray_cell(sat=150)  # delta ≈ 120, above 40
        result = det.detect(cell, "engineer", reference_crop=reference)
        assert result == ["eng_chain_glow"]

    def test_multiple_animations_all_returned(self):
        det = _make_detector()
        det._registry = {
            "alchemist": [
                AnimationEntry("alchemist", "alch_glow", "Alch Glow", "Buff", True, 1.0),
                AnimationEntry("alchemist", "alch_pulse", "Alch Pulse", "Buff", True, 0.5),
            ]
        }
        high_sat = _gray_cell(sat=200)
        result = det.detect(high_sat, "alchemist")
        assert set(result) == {"alch_glow", "alch_pulse"}


# ---------------------------------------------------------------------------
# AnimationDetector.load_from_csv
# ---------------------------------------------------------------------------

class TestLoadFromCSV:

    def _load(self) -> AnimationDetector:
        if not ANIM_CSV.exists():
            pytest.skip("data/Animations.csv not found")
        det = AnimationDetector()
        det.load_from_csv(ANIM_CSV)
        return det

    def test_engineer_loaded(self):
        det = self._load()
        assert "engineer" in det.known_unit_ids()

    def test_engineer_has_animation_id(self):
        det = self._load()
        entries = det._registry.get("engineer", [])
        assert any(e.animation_id == "eng_chain_glow" for e in entries)

    def test_affects_prediction_parsed(self):
        det = self._load()
        entries = det._registry.get("engineer", [])
        assert any(e.affects_prediction is True for e in entries)

    def test_modifier_parsed_as_float(self):
        det = self._load()
        entries = det._registry.get("engineer", [])
        for e in entries:
            assert 0.0 <= e.strength_modifier <= 1.0

    def test_known_unit_ids_nonempty(self):
        det = self._load()
        assert len(det.known_unit_ids()) > 0

    def test_empty_rows_skipped(self):
        """Rows with no Unit ID or Animation ID should not populate registry."""
        det = self._load()
        assert "" not in det.known_unit_ids()


# ---------------------------------------------------------------------------
# AnimationDetector.load_from_db
# ---------------------------------------------------------------------------

class TestLoadFromDB:

    def _make_db_with_animation(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS units (
                unit_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS animations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit_id TEXT NOT NULL REFERENCES units(unit_id),
                animation_name TEXT NOT NULL,
                category TEXT,
                trigger TEXT,
                duration_sec REAL,
                strength_modifier REAL,
                research_status TEXT NOT NULL DEFAULT 'Not Started'
            );
        """)
        conn.execute("INSERT INTO units VALUES ('engineer', 'Engineer')")
        conn.execute(
            "INSERT INTO animations (unit_id, animation_name, category, strength_modifier) "
            "VALUES ('engineer', 'Engineer Chain Glow', 'Intrinsic Buff', 1.0)"
        )
        conn.commit()
        return conn

    def test_entry_loaded_from_db(self):
        conn = self._make_db_with_animation()
        det = AnimationDetector()
        det.load_from_db(conn)
        assert "engineer" in det.known_unit_ids()

    def test_animation_id_derived_from_name(self):
        conn = self._make_db_with_animation()
        det = AnimationDetector()
        det.load_from_db(conn)
        entries = det._registry.get("engineer", [])
        assert any("engineer_chain_glow" in e.animation_id for e in entries)

    def test_affects_prediction_true_when_modifier_nonzero(self):
        conn = self._make_db_with_animation()
        det = AnimationDetector()
        det.load_from_db(conn)
        entries = det._registry.get("engineer", [])
        assert all(e.affects_prediction is True for e in entries)

    def test_null_modifier_treated_as_zero(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE units (unit_id TEXT PRIMARY KEY, display_name TEXT NOT NULL);
            CREATE TABLE animations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit_id TEXT NOT NULL,
                animation_name TEXT NOT NULL,
                category TEXT,
                trigger TEXT,
                duration_sec REAL,
                strength_modifier REAL,
                research_status TEXT NOT NULL DEFAULT 'Not Started'
            );
        """)
        conn.execute("INSERT INTO units VALUES ('archer', 'Archer')")
        conn.execute(
            "INSERT INTO animations (unit_id, animation_name, strength_modifier) "
            "VALUES ('archer', 'Arrow Glow', NULL)"
        )
        conn.commit()
        det = AnimationDetector()
        det.load_from_db(conn)
        entries = det._registry.get("archer", [])
        assert all(e.strength_modifier == 0.0 for e in entries)
        assert all(e.affects_prediction is False for e in entries)

    def test_load_from_db_clears_previous_registry(self):
        conn = self._make_db_with_animation()
        det = AnimationDetector()
        det._registry = {"stale_unit": []}
        det.load_from_db(conn)
        assert "stale_unit" not in det.known_unit_ids()


# ---------------------------------------------------------------------------
# AnimationDetector.known_unit_ids
# ---------------------------------------------------------------------------

class TestKnownUnitIds:

    def test_empty_registry_returns_empty_set(self):
        det = AnimationDetector()
        assert det.known_unit_ids() == set()

    def test_reflects_loaded_entries(self):
        det = _registry_detector()
        assert "engineer" in det.known_unit_ids()

    def test_does_not_include_units_from_cleared_load(self):
        det = _registry_detector()
        det._registry.clear()
        assert det.known_unit_ids() == set()