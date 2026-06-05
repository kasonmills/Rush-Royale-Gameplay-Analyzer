"""
Animation and status-effect detector for board cells in Rush Royale.

Detects active buff/debuff animations using HSV saturation analysis.
Rush Royale buff animations (glows, particle effects) raise the colour
saturation of the affected tile above its baseline appearance.

Two detection modes
-------------------
Absolute (default):
  Cell mean HSV-saturation ≥ sat_threshold  → anomaly detected.

Reference-delta (when reference_crop is supplied):
  Cell mean HSV-saturation − reference mean ≥ delta_threshold → anomaly.

When an anomaly is detected the registry is checked:
  • Unit has registered entries → return their animation_id strings.
  • Unit has no registry data   → return the generic sentinel 'buff_active'.

Usage:
    detector = AnimationDetector()
    detector.load_from_csv("data/Animations.csv")

    animations = detector.detect(cell_crop, "engineer")
    # → ['eng_chain_glow']  when the glow is visible, else []
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HSV saturation channel (0-255) mean above which a cell is considered
# to have an anomalous glow. Rush Royale base art is richly coloured but
# buff particle effects push saturation noticeably higher.
_SAT_ANOMALY_THRESHOLD = 80

# Per-frame saturation increase vs a reference crop required to flag anomaly.
_SAT_DELTA_THRESHOLD = 40

# Crops smaller than this (in pixels²) are skipped — too small to be reliable.
_MIN_CELL_AREA = 100

_MODIFIER_WEIGHTS = {"low": 0.33, "medium": 0.50, "med": 0.50, "high": 1.0}


# ---------------------------------------------------------------------------
# AnimationEntry
# ---------------------------------------------------------------------------

@dataclass
class AnimationEntry:
    """One animation registered for a unit."""
    unit_id: str
    animation_id: str          # short snake_case identifier (from CSV or derived)
    animation_name: str
    category: str              # e.g. 'Intrinsic Buff', 'Debuff'
    affects_prediction: bool   # whether this animation contributes to win score
    strength_modifier: float   # 0.0–1.0 win-prediction weight


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_modifier(raw: str) -> float:
    return _MODIFIER_WEIGHTS.get(raw.strip().lower(), 0.33)


def _to_snake(text: str) -> str:
    """Convert a display name to a snake_case animation_id."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_")


def _mean_saturation(bgr: np.ndarray) -> float:
    """Return mean HSV saturation (0–255) for a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


# ---------------------------------------------------------------------------
# AnimationDetector
# ---------------------------------------------------------------------------

class AnimationDetector:
    """
    Detects active buff/debuff animations on board cell crops.

    Load the registry first with load_from_csv() or load_from_db(), then
    call detect() for each occupied cell.  Cells without a saturation
    anomaly always return an empty list.

    Args:
        sat_threshold:   Absolute HSV-S mean (0–255) that triggers detection.
        delta_threshold: Saturation increase vs reference that triggers detection.
    """

    def __init__(self,
                 sat_threshold: int = _SAT_ANOMALY_THRESHOLD,
                 delta_threshold: int = _SAT_DELTA_THRESHOLD):
        self._registry: dict[str, list[AnimationEntry]] = {}
        self._sat_threshold = sat_threshold
        self._delta_threshold = delta_threshold

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def load_from_csv(self, path: str | Path) -> None:
        """
        Populate the registry from Animations.csv.

        The file has two header rows: the first is a group-label row that
        is skipped; the second contains the actual column names.

        Expected columns (by name):
          Unit ID, Animation ID, Animation Display Name, Category,
          Does it affect Win Prediction?, Modifier Weight (Low/Med/High)
        """
        self._registry.clear()
        with open(Path(path), newline="", encoding="utf-8") as f:
            all_rows = list(csv.reader(f))

        if len(all_rows) < 2:
            return

        fieldnames = all_rows[1]       # row 1 = real column headers
        for raw in all_rows[2:]:       # row 2+ = data
            row = dict(zip(fieldnames, raw))

            unit_id = row.get("Unit ID", "").strip()
            anim_id = row.get("Animation ID", "").strip()
            if not unit_id or not anim_id:
                continue

            affects_raw = row.get("Does it affect Win Prediction?", "").strip().lower()
            affects = affects_raw in ("yes", "true", "1")

            modifier_raw = row.get("Modifier Weight (Low/Med/High)", "").strip()
            modifier = _parse_modifier(modifier_raw)

            entry = AnimationEntry(
                unit_id=unit_id,
                animation_id=anim_id,
                animation_name=row.get("Animation Display Name", "").strip(),
                category=row.get("Category", "").strip(),
                affects_prediction=affects,
                strength_modifier=modifier,
            )
            self._registry.setdefault(unit_id, []).append(entry)

    def load_from_db(self, conn) -> None:
        """
        Populate the registry from the animations table in unit_meta.db.
        Clears any previously loaded entries.

        animation_id is derived from animation_name (snake_case).
        affects_prediction is True when strength_modifier > 0.
        """
        self._registry.clear()
        rows = conn.execute(
            """
            SELECT unit_id, animation_name, category, strength_modifier
            FROM animations
            """
        ).fetchall()
        for row in rows:
            sm = row["strength_modifier"]
            try:
                modifier = float(sm) if sm is not None else 0.0
            except (TypeError, ValueError):
                modifier = _parse_modifier(str(sm))

            anim_name = row["animation_name"] or ""
            anim_id = _to_snake(anim_name) if anim_name else "buff_active"

            entry = AnimationEntry(
                unit_id=row["unit_id"],
                animation_id=anim_id,
                animation_name=anim_name,
                category=row["category"] or "",
                affects_prediction=modifier > 0.0,
                strength_modifier=modifier,
            )
            self._registry.setdefault(entry.unit_id, []).append(entry)

    def known_unit_ids(self) -> set[str]:
        """Return the set of unit_ids with at least one registered animation."""
        return set(self._registry.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self,
               cell_crop: np.ndarray,
               unit_id: str,
               reference_crop: Optional[np.ndarray] = None) -> list[str]:
        """
        Return animation IDs detected as active on this cell crop.

        Args:
            cell_crop:      BGR image of a single board cell.
            unit_id:        The unit occupying the cell.
            reference_crop: Optional baseline crop (same unit, no animation).
                            When provided, delta-based detection is used.

        Returns:
            List of animation_id strings for active animations.
            ['buff_active'] when an anomaly is found but no registry entry exists.
            [] when no anomaly is detected or the crop is too small.
        """
        if cell_crop.size < _MIN_CELL_AREA:
            return []

        if not self._is_anomalous(cell_crop, reference_crop):
            return []

        entries = self._registry.get(unit_id, [])
        if not entries:
            return ["buff_active"]

        return [e.animation_id for e in entries]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_anomalous(self,
                      cell_crop: np.ndarray,
                      reference_crop: Optional[np.ndarray]) -> bool:
        """
        Return True when the cell's colour saturation indicates a buff glow.

        Reference mode: saturation delta ≥ _delta_threshold.
        Absolute mode:  saturation mean  ≥ _sat_threshold.
        """
        cell_sat = _mean_saturation(cell_crop)

        if reference_crop is not None and reference_crop.size >= _MIN_CELL_AREA:
            ref_sat = _mean_saturation(reference_crop)
            return (cell_sat - ref_sat) >= self._delta_threshold

        return cell_sat >= self._sat_threshold