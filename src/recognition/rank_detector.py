"""
Shape-based merge rank detector.

Rush Royale tile borders change shape with merge rank:
  Rank 1 — circle
  Rank 2 — football (elongated oval)
  Rank 3 — triangle
  Rank 4 — square
  Rank 5 — pentagon
  Rank 6 — hexagon
  Rank 7 — heptagon

This module detects rank from the tile border contour using
cv2.approxPolyDP — no reference images or training data required.

Strategy:
  1. Convert cell crop to grayscale and threshold to isolate the
     bright tile border against the darker game background.
  2. Find the largest external contour (the border itself).
  3. Approximate the contour to a polygon and count sides.
  4. Map side count → rank, with special handling for circles
     (circularity ratio) and footballs (elongation ratio).

Usage:
    detector = RankDetector()
    rank = detector.detect(cell_crop)
    if rank is not None:
        print(f"Rank {rank}")   # 1–7, or None if detection failed
"""

from typing import Optional

import cv2
import numpy as np


# Minimum contour area as a fraction of cell area.
# Filters noise; the border should occupy a significant portion of the cell.
_MIN_CONTOUR_AREA_FRAC = 0.10

# Approximation epsilon as a fraction of contour arc length.
# Higher = coarser polygon; lower = finer (more sides detected).
_APPROX_EPSILON_FRAC = 0.04

# Circularity threshold for rank-1 circle detection.
# circularity = 4π·area / perimeter²; perfect circle = 1.0
_CIRCLE_CIRCULARITY_MIN = 0.70

# Elongation (minor/major axis ratio) threshold for rank-2 football detection.
# A football is more elongated than a circle but less circular overall.
_FOOTBALL_MAX_CIRCULARITY = 0.70   # below circle threshold
_FOOTBALL_MIN_CIRCULARITY = 0.45   # above random noise

# Polygon side count → rank mapping (after approxPolyDP).
_SIDES_TO_RANK: dict[int, int] = {
    3: 3,  # triangle
    4: 4,  # square
    5: 5,  # pentagon
    6: 6,  # hexagon
    7: 7,  # heptagon
}


class RankDetector:
    """
    Detects merge rank from a single board cell crop.

    Returns None when the cell is empty or the border contour cannot be
    cleanly identified (low contrast, partial crop, animated tile, etc.).
    In those cases the caller should fall back to the template matcher's
    rank field or leave rank as unknown for this frame.
    """

    def detect(self, cell_crop: np.ndarray) -> Optional[int]:
        """
        Detect the merge rank from a cell crop.

        Args:
            cell_crop: BGR image of a single board cell (any size).

        Returns:
            Integer rank 1–7, or None if detection is inconclusive.
        """
        if cell_crop is None or cell_crop.size == 0:
            return None

        contour = self._find_border_contour(cell_crop)
        if contour is None:
            return None

        return self._classify_contour(contour, cell_crop.shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_border_contour(
            self, cell_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Isolates the tile border and returns its contour, or None.

        The tile border is the brightest, most distinct shape in the crop.
        We threshold the top brightness percentile and find the largest
        external contour.
        """
        cell_h, cell_w = cell_crop.shape[:2]
        min_area = cell_h * cell_w * _MIN_CONTOUR_AREA_FRAC

        gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold — handles both bright gold borders and
        # dimmer blue/silver borders from different unit appearances.
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Small morph close to fill gaps in the border line itself.
        k = max(3, min(cell_w, cell_h) // 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick the largest contour that meets the minimum area threshold.
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area:
            return None

        return best

    def _classify_contour(
            self,
            contour: np.ndarray,
            cell_shape: tuple) -> Optional[int]:
        """
        Maps a contour to a rank 1–7 using shape metrics.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 1:
            return None

        # Circularity: 1.0 = perfect circle, lower = more angular/elongated.
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # Rank 1 — circle
        if circularity >= _CIRCLE_CIRCULARITY_MIN:
            return 1

        # Rank 2 — football (elongated oval).
        # Distinguish from a near-circle by checking aspect ratio via
        # the fitted ellipse (available when contour has ≥ 5 points).
        if (_FOOTBALL_MIN_CIRCULARITY <= circularity < _FOOTBALL_MAX_CIRCULARITY
                and len(contour) >= 5):
            _, (ma, mb), _ = cv2.fitEllipse(contour)
            if ma > 0 and mb > 0:
                minor, major = sorted((ma, mb))
                elongation = minor / major  # 1.0 = circle, lower = more elongated
                if elongation < 0.75:       # clearly not a circle
                    return 2

        # Ranks 3–7 — polygons.
        epsilon = _APPROX_EPSILON_FRAC * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        sides = len(approx)

        # Try progressively looser approximation if side count is too high
        # (noisy contour from a low-res crop).
        if sides > 7:
            for factor in (0.06, 0.08, 0.10):
                approx = cv2.approxPolyDP(contour, factor * perimeter, True)
                sides = len(approx)
                if sides <= 7:
                    break

        return _SIDES_TO_RANK.get(sides)