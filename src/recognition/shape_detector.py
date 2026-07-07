"""
Merge-rank shape detector for Rush Royale board cells.

Each merge rank has a distinctive polygon border around the unit sprite:
  Rank 1 → Circle
  Rank 2 → Oval / football shape
  Rank 3 → Triangle
  Rank 4 → Diamond (rotated square)
  Rank 5 → Pentagon
  Rank 6 → Hexagon
  Rank 7 → Heptagon

detect_merge_rank() uses edge detection and contour approximation to identify
which shape is present in a cell crop, then returns a clean programmatic mask
for that shape.  The mask is used to restrict NCC comparison to only the pixels
that fall inside the shape — ignoring background grid, border decorations, and
any UI effects that surround the unit sprite.

When shape detection fails (empty cell, animation occlusion, etc.) the function
returns rank=0 and a conservative circular fallback mask that still removes most
corner background noise even without knowing the exact shape.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Polygon mask radius as a fraction of the cell's smaller half-dimension.
# 0.42 fills most of the cell while leaving a small margin away from the
# actual shape border (so border decoration pixels are excluded).
_MASK_RADIUS_FRAC = 0.42

# Oval axes multipliers relative to the base radius.
_OVAL_X_FRAC = 1.25
_OVAL_Y_FRAC = 0.80

# Contour area bounds relative to cell area.  Too small → noise artefact;
# too large → full-cell bounding box rather than the shape itself.
_MIN_AREA_FRAC = 0.12
_MAX_AREA_FRAC = 0.92

# approxPolyDP epsilon as a fraction of the contour perimeter.
# At 0.020, the approximation is tight enough that:
#   - Triangle through heptagon yield exactly 3–7 vertices (straight edges fit cleanly)
#   - Circle/oval yield ≥ 8 vertices (curved arcs resist collapsing further)
# This makes vertex count alone a reliable discriminator across all 7 ranks.
_APPROX_EPSILON = 0.020

# Vertex count at or above which a contour is classified as a curved shape
# (circle or oval) rather than a polygon.  Empirically, filled-frame circles
# produce 8 vertices while heptagons produce 7 with the epsilon above.
_CIRCLE_MIN_VERTS = 8

# Bounding-box aspect ratio threshold for distinguishing oval (rank 2) from
# circle (rank 1) once the vertex count indicates a round shape.
_OVAL_ASPECT_RATIO = 1.15

# Fallback mask radius fraction used when rank detection fails entirely.
_FALLBACK_RADIUS_FRAC = 0.38


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_merge_rank(crop: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Detect the merge-rank polygon border in a cell crop.

    Returns:
        (rank, mask)
          rank: 1–7 for the detected shape, or 0 if detection failed.
          mask: uint8 binary image — 255 inside the shape, 0 outside.
                When rank=0 a conservative circular fallback mask is returned.
    """
    h, w = crop.shape[:2]
    rank = _classify_shape(crop)
    mask = generate_rank_mask(rank, h, w)
    return rank, mask


def generate_rank_mask(rank: int, h: int, w: int) -> np.ndarray:
    """
    Generate a clean binary mask for the given merge-rank shape.

    Args:
        rank: 1–7 for a specific shape; 0 for the circular fallback.
        h, w: Target mask height and width in pixels.

    Returns:
        uint8 ndarray of shape (h, w): 255 inside the polygon, 0 outside.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * _MASK_RADIUS_FRAC)

    if rank == 1:
        cv2.circle(mask, (cx, cy), r, 255, -1)

    elif rank == 2:
        axes = (int(r * _OVAL_X_FRAC), int(r * _OVAL_Y_FRAC))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    elif 3 <= rank <= 7:
        pts = _regular_polygon_pts(rank, cx, cy, r)
        cv2.fillConvexPoly(mask, pts, 255)

    else:
        # Fallback: slightly smaller circle that still covers the centre sprite
        cv2.circle(mask, (cx, cy), int(min(h, w) * _FALLBACK_RADIUS_FRAC), 255, -1)

    return mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_shape(crop: np.ndarray) -> int:
    """
    Classify the merge-rank polygon in the crop.
    Returns rank 1–7, or 0 on detection failure.
    """
    h, w = crop.shape[:2]
    cell_area = float(h * w)

    # ---- Preprocessing ------------------------------------------------
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 90)

    # Close small gaps in the shape border so the contour is connected
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # ---- Contour detection --------------------------------------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    # Keep only contours whose area falls in a plausible range
    valid = [c for c in contours
             if _MIN_AREA_FRAC * cell_area
                < cv2.contourArea(c)
                < _MAX_AREA_FRAC * cell_area]
    if not valid:
        return 0

    # Pick the largest candidate — it should be the rank shape border
    cnt  = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri < 1.0:
        return 0

    # ---- Shape classification -----------------------------------------
    # With epsilon=0.020×perimeter, approxPolyDP yields vertex counts that
    # cleanly separate all 7 ranks: straight-edged polygons (3–7 vertices)
    # vs. curved shapes (8+ vertices for circle/oval).
    approx  = cv2.approxPolyDP(cnt, _APPROX_EPSILON * peri, True)
    n_verts = len(approx)

    if n_verts >= _CIRCLE_MIN_VERTS:
        # Curved shape: distinguish circle from oval by bounding-box aspect ratio
        _x, _y, bw, bh = cv2.boundingRect(cnt)
        aspect = (bw / bh) if bh > 0 else 1.0
        return 2 if aspect >= _OVAL_ASPECT_RATIO else 1

    if n_verts == 3:
        return 3
    if n_verts == 4:
        return 4
    if n_verts == 5:
        return 5
    if n_verts == 6:
        return 6
    if n_verts == 7:
        return 7

    return 0


def _regular_polygon_pts(n: int, cx: int, cy: int, r: int) -> np.ndarray:
    """
    Vertices of a regular n-gon centred at (cx, cy) with circumradius r.
    The first vertex points straight up (angle offset = -π/2) so that:
      n=3 → upward-pointing triangle
      n=4 → diamond (vertex at top/bottom/left/right)
      n=5..7 → flat-bottomed standard orientations
    """
    pts = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n - math.pi / 2.0
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)