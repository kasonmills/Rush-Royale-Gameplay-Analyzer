"""
Tests for UnitClassifier.

ultralytics is always mocked so tests run without a trained model or GPU.
The _make_classifier helper directly injects a mock YOLO model rather than
patching the import, keeping the tests independent of how ultralytics is
installed.

Covers:
  - UnitClassifier.available: False before load, True after mock-load
  - UnitClassifier.load(): FileNotFoundError for missing model
  - UnitClassifier.classify(): None when unavailable, None for empty cell,
      None below threshold, ClassifyResult when confident,
      deck-constraint respected (top-1 in candidates, top-1 not in candidates),
      fallback to best candidate when top-1 excluded
  - UnitClassifier.known_unit_ids(): reflects loaded class names
  - CLASSIFY_THRESHOLD constant exposed and usable
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.recognition.unit_classifier import (
    CLASSIFY_THRESHOLD,
    ClassifyResult,
    UnitClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_cell(h: int = 100, w: int = 100,
                val: int = 128) -> np.ndarray:
    """Non-empty cell with enough mean + std to pass the empty-cell check."""
    arr = np.full((h, w, 3), val, dtype=np.uint8)
    arr[::2, ::2] = max(0, val - 20)  # chequerboard so std_dev > 0
    return arr


def _empty_cell() -> np.ndarray:
    """Cell that _is_empty_cell() flags as empty (dark + uniform)."""
    return np.full((100, 100, 3), 5, dtype=np.uint8)


def _make_mock_model(names: dict[int, str],
                     top1_idx: int,
                     top1_conf: float,
                     all_confs: dict[int, float] | None = None) -> MagicMock:
    """
    Build a mock YOLO model whose __call__ returns inference-like output.

    all_confs: per-class softmax probabilities {class_idx: prob}.
               Defaults to top1_conf for top1_idx, 0.0 elsewhere.
    """
    if all_confs is None:
        all_confs = {i: 0.0 for i in names}
        all_confs[top1_idx] = top1_conf

    probs = MagicMock()
    probs.top1     = top1_idx
    probs.top1conf = top1_conf
    probs.data     = np.array(
        [all_confs.get(i, 0.0) for i in range(len(names))], dtype=np.float32
    )

    result = MagicMock()
    result.probs = probs

    model = MagicMock()
    model.names = names
    model.return_value = [result]
    return model


def _make_classifier(names: dict[int, str],
                     top1_idx: int,
                     top1_conf: float,
                     all_confs: dict[int, float] | None = None,
                     threshold: float = CLASSIFY_THRESHOLD) -> UnitClassifier:
    """Create a UnitClassifier with a directly-injected mock model."""
    clf = UnitClassifier(threshold=threshold)
    clf._model = _make_mock_model(names, top1_idx, top1_conf, all_confs)
    clf._names = names
    return clf


# ---------------------------------------------------------------------------
# available
# ---------------------------------------------------------------------------

class TestAvailable:

    def test_false_before_load(self):
        clf = UnitClassifier()
        assert clf.available is False

    def test_true_after_mock_injected(self):
        clf = _make_classifier({0: "alchemist", 1: "archer"}, 0, 0.9)
        assert clf.available is True


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:

    def test_raises_file_not_found_for_missing_model(self):
        """load() must raise FileNotFoundError when the .pt file does not exist."""
        clf = UnitClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load("nonexistent_model_that_does_not_exist.pt")


# ---------------------------------------------------------------------------
# classify() — basic paths
# ---------------------------------------------------------------------------

class TestClassifyBasic:

    def test_returns_none_when_unavailable(self):
        clf = UnitClassifier()
        assert clf.classify(_solid_cell()) is None

    def test_returns_none_for_zero_size_crop(self):
        clf = _make_classifier({0: "alchemist"}, 0, 0.9)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert clf.classify(empty) is None

    def test_returns_none_for_empty_cell(self):
        clf = _make_classifier({0: "alchemist"}, 0, 0.9)
        assert clf.classify(_empty_cell()) is None

    def test_returns_none_below_threshold(self):
        clf = _make_classifier({0: "alchemist"}, 0, 0.3, threshold=0.5)
        assert clf.classify(_solid_cell()) is None

    def test_returns_result_above_threshold(self):
        clf = _make_classifier({0: "alchemist"}, 0, 0.9)
        result = clf.classify(_solid_cell())
        assert isinstance(result, ClassifyResult)
        assert result.unit_id == "alchemist"
        assert result.confidence == pytest.approx(0.9)

    def test_at_threshold_boundary(self):
        """Confidence exactly at threshold should be accepted."""
        clf = _make_classifier({0: "alchemist"}, 0, 0.5, threshold=0.5)
        result = clf.classify(_solid_cell())
        assert result is not None

    def test_unit_id_matches_model_names(self):
        names = {0: "archer", 1: "rogue", 2: "knight"}
        clf = _make_classifier(names, 2, 0.85)
        result = clf.classify(_solid_cell())
        assert result.unit_id == "knight"


# ---------------------------------------------------------------------------
# classify() — deck constraint (candidates)
# ---------------------------------------------------------------------------

class TestClassifyWithCandidates:

    def test_top1_in_candidates_returned(self):
        names = {0: "alchemist", 1: "archer", 2: "rogue"}
        clf = _make_classifier(names, 1, 0.88)
        result = clf.classify(_solid_cell(), candidates={"archer", "rogue"})
        assert result is not None
        assert result.unit_id == "archer"

    def test_top1_not_in_candidates_falls_back(self):
        """Top-1 is 'alchemist' (not in candidates). Classifier falls back to
        the best allowed candidate. Threshold is lowered so the fallback conf
        of 0.20 clears the bar."""
        names = {0: "alchemist", 1: "archer", 2: "rogue"}
        all_confs = {0: 0.70, 1: 0.20, 2: 0.10}
        clf = _make_classifier(names, 0, 0.70, all_confs=all_confs, threshold=0.10)
        result = clf.classify(_solid_cell(), candidates={"archer", "rogue"})
        # Best allowed candidate is archer (conf=0.20 > threshold=0.10)
        assert result is not None
        assert result.unit_id == "archer"
        assert result.confidence == pytest.approx(0.20)

    def test_fallback_returns_none_when_below_threshold(self):
        """All candidates have confidence below threshold → None."""
        names = {0: "alchemist", 1: "archer", 2: "rogue"}
        all_confs = {0: 0.80, 1: 0.05, 2: 0.03}
        clf = _make_classifier(names, 0, 0.80, all_confs=all_confs, threshold=0.5)
        result = clf.classify(_solid_cell(), candidates={"archer", "rogue"})
        assert result is None

    def test_empty_candidates_returns_none(self):
        names = {0: "alchemist", 1: "archer"}
        clf = _make_classifier(names, 0, 0.9)
        result = clf.classify(_solid_cell(), candidates=set())
        assert result is None

    def test_single_candidate_that_is_top1(self):
        names = {0: "alchemist"}
        clf = _make_classifier(names, 0, 0.95)
        result = clf.classify(_solid_cell(), candidates={"alchemist"})
        assert result is not None
        assert result.unit_id == "alchemist"


# ---------------------------------------------------------------------------
# known_unit_ids()
# ---------------------------------------------------------------------------

class TestKnownUnitIds:

    def test_empty_before_load(self):
        clf = UnitClassifier()
        assert clf.known_unit_ids() == set()

    def test_reflects_model_names(self):
        names = {0: "alchemist", 1: "archer", 2: "rogue"}
        clf = _make_classifier(names, 0, 0.9)
        assert clf.known_unit_ids() == {"alchemist", "archer", "rogue"}


# ---------------------------------------------------------------------------
# CLASSIFY_THRESHOLD constant
# ---------------------------------------------------------------------------

class TestConstants:

    def test_threshold_is_float(self):
        assert isinstance(CLASSIFY_THRESHOLD, float)

    def test_threshold_in_range(self):
        assert 0.0 < CLASSIFY_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# Integration: classifier confidence override in MCR context
# ---------------------------------------------------------------------------

class TestClassifierConfidenceOverride:
    """
    Verify that when the classifier is more confident than NCC it would
    override — test the logic in isolation (MCR wiring tested by running the
    full pipeline, not mocked here).
    """

    def test_higher_classifier_conf_wins(self):
        names = {0: "engineer", 1: "harlequin"}
        # Template matcher found 'harlequin' at conf 0.65; classifier sees 'engineer' at 0.92
        clf = _make_classifier(names, 0, 0.92)
        result = clf.classify(_solid_cell(), candidates={"engineer", "harlequin"})
        assert result.unit_id == "engineer"
        assert result.confidence > 0.65

    def test_lower_classifier_conf_does_not_win(self):
        """Caller (MCR) should ignore classifier when its conf < NCC — verify
        the returned confidence so MCR can make that decision."""
        names = {0: "engineer", 1: "harlequin"}
        clf = _make_classifier(names, 0, 0.55)
        result = clf.classify(_solid_cell(), candidates={"engineer", "harlequin"})
        # Classifier returns a result (0.55 >= 0.50 threshold), but with conf < NCC (0.65)
        # MCR would keep the template matcher result in that case
        assert result is not None
        assert result.confidence < 0.65