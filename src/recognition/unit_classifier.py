"""
YOLOv8 classification-based unit recogniser for board cells.

Identifies unit type from a board cell crop using a trained YOLOv8-cls model.
Designed to complement TemplateMatcher: when the classifier's top-1 confidence
exceeds the template matcher's NCC score, the classifier's unit_id wins.

Merge rank and appearance state are NOT determined by this classifier — the
current model is trained on cell crops and labels unit_id only.  Rank
detection still relies on TemplateMatcher once the unit_id is known.

When no model file is present (available == False), all public methods return
None and callers fall back to TemplateMatcher unchanged.

Model training
--------------
See tools/prepare_dataset.py (build synthetic training set from reference
images) and tools/train_classifier.py (run YOLOv8n-cls training).  The
trained model is saved to assets/models/unit_classifier.pt.

Adding new units or ranks
--------------------------
1. Add reference image(s) to assets/reference/<unit_id>/
2. Re-run tools/prepare_dataset.py
3. Re-run tools/train_classifier.py
No code changes required.

Usage:
    clf = UnitClassifier()
    clf.load("assets/models/unit_classifier.pt")
    if clf.available:
        result = clf.classify(cell_crop, candidates=player_deck)
        # → ClassifyResult(unit_id='alchemist', confidence=0.94)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    _YOLO = None  # type: ignore[assignment]

# Minimum top-1 softmax probability to accept a prediction.
CLASSIFY_THRESHOLD = 0.50

# Input size fed to the model. Must match the --imgsz used during training.
_INPUT_SIZE = 128

# Cells with mean pixel value below this are treated as empty.
_EMPTY_MEAN_THRESHOLD = 15
# Cells with pixel std-dev below this are treated as empty (blank/uniform).
_EMPTY_STD_THRESHOLD = 8


@dataclass
class ClassifyResult:
    """Result from UnitClassifier.classify()."""
    unit_id: str
    confidence: float


def _is_empty_cell(cell_crop: np.ndarray) -> bool:
    if cell_crop.size == 0:
        return True
    gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
    mean_val, std_dev = cv2.meanStdDev(gray)
    return float(mean_val[0][0]) < _EMPTY_MEAN_THRESHOLD and \
           float(std_dev[0][0]) < _EMPTY_STD_THRESHOLD


class UnitClassifier:
    """
    YOLOv8 classification wrapper for board cell unit identification.

    Load once with load(); then call classify() per cell.  All methods
    are safe to call when the model is not loaded — they return None.

    Args:
        threshold: Minimum top-1 confidence to accept a classification.
                   Below this the result is treated as 'no match'.
    """

    def __init__(self, threshold: float = CLASSIFY_THRESHOLD):
        self._model = None
        self._names: dict[int, str] = {}  # class index → unit_id
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """
        Load a trained YOLOv8 classification model.

        Args:
            path: Path to the .pt model file (e.g. assets/models/unit_classifier.pt).

        Raises:
            FileNotFoundError: If the model file does not exist.
            ImportError:       If ultralytics is not installed.
        """
        if _YOLO is None:
            raise ImportError(
                "ultralytics is required to use UnitClassifier. "
                "Install it with: pip install ultralytics"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model = _YOLO(str(path))
        self._names = dict(self._model.names or {})

    @property
    def available(self) -> bool:
        """True when a model is loaded and ready for inference."""
        return self._model is not None

    def known_unit_ids(self) -> set[str]:
        """Return the set of unit_ids the model was trained on."""
        return set(self._names.values())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self,
                 cell_crop: np.ndarray,
                 candidates: Optional[set[str]] = None) -> Optional[ClassifyResult]:
        """
        Classify a board cell crop and return the most likely unit_id.

        Args:
            cell_crop:  BGR image of a single board cell (any size — resized
                        internally to _INPUT_SIZE × _INPUT_SIZE).
            candidates: Deck-constrained set of unit_ids to consider.  When
                        provided, the top-1 prediction is accepted only if it
                        belongs to this set; otherwise the highest-confidence
                        candidate from the set is returned.  Pass None to
                        allow any unit the model knows.

        Returns:
            ClassifyResult, or None when:
              - The model is not loaded.
              - The cell appears empty.
              - No prediction meets the confidence threshold.
              - candidates is provided and no candidate unit is confident enough.
        """
        if not self.available or cell_crop.size == 0:
            return None
        if _is_empty_cell(cell_crop):
            return None

        resized = cv2.resize(cell_crop, (_INPUT_SIZE, _INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        results = self._model(resized, verbose=False)
        if not results:
            return None

        probs = results[0].probs
        if probs is None:
            return None

        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        top1_id = self._names.get(top1_idx, "")

        if candidates is None:
            if not top1_id or top1_conf < self._threshold:
                return None
            return ClassifyResult(unit_id=top1_id, confidence=top1_conf)

        # Deck-constrained: top-1 must be in candidates, else search candidates.
        if top1_id in candidates and top1_conf >= self._threshold:
            return ClassifyResult(unit_id=top1_id, confidence=top1_conf)

        best_id, best_conf = self._best_candidate(probs, candidates)
        if best_id is None or best_conf < self._threshold:
            return None
        return ClassifyResult(unit_id=best_id, confidence=best_conf)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _best_candidate(self,
                        probs,
                        candidates: set[str]) -> tuple[Optional[str], float]:
        """Return (unit_id, confidence) for the highest-confidence candidate."""
        best_conf = -1.0
        best_id: Optional[str] = None
        for idx, name in self._names.items():
            if name in candidates:
                conf = float(probs.data[idx])
                if conf > best_conf:
                    best_conf = conf
                    best_id = name
        return best_id, best_conf