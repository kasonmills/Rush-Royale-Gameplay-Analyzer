"""
Talent icon classifier — identifies the talent tier and branch from the badge
overlaid on a board cell.

Talent rules:
  - Talents are cumulative. A unit displaying a T3 badge has T1, T2, and T3
    all active simultaneously. Seeing T2 means T1 is also active.
  - Each tier is XOR: either a branching tier (player chose L or R) OR a fixed
    tier (no choice — one path exists). Which type each tier is depends on the
    unit (stored in talent_trees in the DB).

What this classifier does:
  - Reads the DISPLAYED badge (the highest active tier) from a cell crop.
  - Returns the observed tier and that tier's branch.
  - The caller is responsible for building the full talent_path in UnitCell:
      * All tiers 1..observed_tier are marked active.
      * Branches for tiers below the observed tier are filled from DB metadata
        (Fixed tiers are known; branching tiers require historical observation
        or MCR tracking).

Reference badge images live in:
  assets/reference/talent_icons/
    1_L.png, 1_R.png, 1_Fixed.png  ← T1 badge variants
    2_L.png, 2_R.png, 2_Fixed.png  ← T2 badge variants
    ...
    4_L.png, 4_R.png, 4_Fixed.png  ← T4 badge variants

Usage:
    classifier = TalentClassifier()
    classifier.load("assets/reference/talent_icons")

    result = classifier.classify(cell_crop)
    if result:
        # result.tier = highest badge visible (e.g. 3)
        # result.branch = branch chosen for THAT tier (e.g. 'L')
        # Tiers 1..result.tier-1 are also active; their branches come from the DB
        print(result.tier, result.branch, result.confidence)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.55

# Badge sits in the top-left corner of the cell.
# Fractions are relative to the cell's own width/height.
BADGE_WIDTH_FRAC  = 0.35
BADGE_HEIGHT_FRAC = 0.30

VALID_TIERS    = (1, 2, 3, 4)
VALID_BRANCHES = ("L", "R", "Fixed")  # each tier uses exactly one of these


@dataclass
class TalentResult:
    """
    The talent badge detected on a board cell.

    tier:       The highest active tier displayed on the badge (1–4).
                All tiers 1 through `tier` are therefore active.
    branch:     The branch chosen FOR THIS TIER ('L', 'R', or 'Fixed').
                Branches for lower tiers are not encoded in this result;
                they must be resolved separately via DB lookup or MCR tracking.
    confidence: NCC match score (0.0–1.0).
    """
    tier: int
    branch: str
    confidence: float


class TalentClassifier:
    """
    Matches cell badge regions against reference talent icon images.

    Args:
        threshold: NCC score below which no talent badge is reported.
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        # (tier, branch) → [full-size image, half-size image]
        self._templates: dict[tuple[int, str], list[np.ndarray]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, icon_dir: str | Path):
        """
        Load talent badge reference images from icon_dir.

        Expected filenames: <tier>_<branch>.png  (e.g. 1_L.png, 4_Fixed.png)
        Missing files are silently skipped — only loaded combinations are matched.
        """
        icon_dir = Path(icon_dir)
        self._templates.clear()

        for tier in VALID_TIERS:
            for branch in VALID_BRANCHES:
                path = icon_dir / f"{tier}_{branch}.png"
                if not path.exists():
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                self._templates[(tier, branch)] = [
                    img,
                    cv2.resize(img, (max(1, w // 2), max(1, h // 2)),
                               interpolation=cv2.INTER_AREA),
                ]

        self._loaded = True
        print(f"[TalentClassifier] Loaded {len(self._templates)} talent badge "
              f"templates from {icon_dir}")

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, cell_crop: np.ndarray) -> Optional[TalentResult]:
        """
        Classify the talent badge in a board cell crop.

        Returns TalentResult if a badge is detected above the confidence
        threshold, or None if no badge is visible (no talent active on
        this cell).
        """
        if not self._loaded or cell_crop.size == 0:
            return None

        badge = _extract_badge(cell_crop)
        if badge is None or badge.size == 0:
            return None

        best_score = -1.0
        best_key: Optional[tuple[int, str]] = None

        for (tier, branch), scaled_imgs in self._templates.items():
            score = _best_ncc(badge, scaled_imgs)
            if score > best_score:
                best_score = score
                best_key = (tier, branch)

        if best_score < self.threshold or best_key is None:
            return None

        return TalentResult(
            tier=best_key[0],
            branch=best_key[1],
            confidence=best_score,
        )

    def classify_all(self,
                     cell_crops: list[tuple[str, int, int, np.ndarray]]
                     ) -> dict[tuple[str, int, int], Optional[TalentResult]]:
        """
        Batch classify all cells from GridCalibrator.all_cell_crops().
        Returns {(player, row, col): TalentResult | None}.
        """
        return {
            (player, row, col): self.classify(crop)
            for player, row, col, crop in cell_crops
        }

    def build_talent_path(self,
                          result: TalentResult,
                          unit_id: str,
                          db_conn) -> dict[int, Optional[str]]:
        """
        Build the full cumulative talent_path dict for a UnitCell from a
        single TalentResult, using the DB to fill in lower-tier branches.

        For each tier below result.tier:
          - If the DB says it's 'Fixed', the branch is known without observation.
          - If it's a branching tier (L/R), the branch is marked None until
            the MCR accumulates enough observations.

        Args:
            result:  The TalentResult from classify().
            unit_id: The unit's ID (used to query talent_trees).
            db_conn: An open sqlite3 connection to unit_meta.db.

        Returns:
            dict mapping tier → branch for all tiers 1..result.tier.
        """
        from src.database.unit_meta_repo import TalentRepo

        path: dict[int, Optional[str]] = {}

        for tier in range(1, result.tier + 1):
            if tier == result.tier:
                path[tier] = result.branch
                continue
            # Look up this tier's branch type from the DB
            row = TalentRepo.get_branch(db_conn, unit_id, tier, "Fixed")
            if row is not None:
                path[tier] = "Fixed"
            else:
                # Branching tier — branch unknown until observed directly
                path[tier] = None

        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_badge(cell_crop: np.ndarray) -> Optional[np.ndarray]:
    """Crops the badge region from the top-left corner of the cell."""
    h, w = cell_crop.shape[:2]
    if h < 16 or w < 16:
        return None
    x2 = max(8, int(w * BADGE_WIDTH_FRAC))
    y2 = max(8, int(h * BADGE_HEIGHT_FRAC))
    return cell_crop[0:y2, 0:x2]


def _best_ncc(query: np.ndarray, templates: list[np.ndarray]) -> float:
    """Best NCC score across a list of template scales."""
    best = -1.0
    for tmpl in templates:
        th, tw = tmpl.shape[:2]
        resized = cv2.resize(query, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
    return best
