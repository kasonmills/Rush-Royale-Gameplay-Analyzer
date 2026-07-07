"""
BoardEffectClassifier — identifies artifact activations and hero skill effects
that appear as visual overlays on board cells during gameplay.

Rush Royale PvP produces two categories of in-cell visual effects:
  - Hero skill effects: distinctive glows, marks, or overlays placed on cells
    when a hero activates their skill (e.g. Zeus lightning mark, Necromancer
    resurrection aura).
  - Artifact effects: visual indicators on cells triggered by an equipped
    artifact (e.g. a charged cell glow, a buff overlay).

Reference images live in:
  assets/reference/artifacts/<artifact_id>/<visual_state>.png
  assets/reference/hero_board_effects/<hero_id>/<effect_name>.png

Naming conventions:
  <artifact_id>   — matches artifact_id in unit_meta.db artifacts table
  <hero_id>       — matches hero_id in unit_meta.db heroes table
  <visual_state>  — e.g. 'active', 'cooldown', 'charged'
  <effect_name>   — e.g. 'skill_mark', 'passive_glow', 'resurrection_zone'

Usage:
    clf = BoardEffectClassifier()
    clf.load("assets/reference")

    result = clf.classify(cell_crop)
    if result:
        print(result.effect_type, result.effect_id, result.confidence)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Strips trailing variant numbers (-2, -3 …) before the _tile suffix check.
# wind_tile-2 → wind_tile  |  sacrifice_tile-2 → sacrifice_tile
_VARIANT_SUFFIX_RE = re.compile(r"-\d+$")

# Folders inside assets/reference/ that are NOT unit folders.
_NON_UNIT_DIRS = frozenset({
    "hero_portraits", "talent_icons", "artifacts", "hero_board_effects",
})


def _is_tile_stem(stem: str) -> bool:
    """True when the stem (after stripping -N variant suffix) ends with _tile."""
    return _VARIANT_SUFFIX_RE.sub("", stem).endswith("_tile")


def _is_rankless(stem: str) -> bool:
    """True when the stem contains no 'rank' token.

    Files without a rank token are skipped by TemplateMatcher's filename
    parser (e.g. kobold_mine, kobold_gold_mine).  If they live in a unit
    folder they are assumed to represent placed-object tiles that should
    count as empty cells for filtering purposes.
    """
    return "rank" not in stem


CONFIDENCE_THRESHOLD = 0.55


@dataclass
class BoardEffect:
    """
    A detected visual effect on a board cell.

    effect_type:  'artifact' or 'hero_skill'
    effect_id:    artifact_id or hero_id from the reference folder name
    effect_name:  visual_state / effect_name from the reference image filename
    confidence:   NCC match score (0.0–1.0)
    """
    effect_type: str
    effect_id:   str
    effect_name: str
    confidence:  float


@dataclass
class _EffectTemplate:
    effect_type: str
    effect_id:   str
    effect_name: str
    images: list  # [full-size BGR, half-size BGR]


class BoardEffectClassifier:
    """
    NCC-based classifier for artifact and hero skill visual effects on board cells.

    Args:
        threshold: NCC score below which no effect is reported (default 0.55).
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self._templates: list[_EffectTemplate] = []
        self._empty_tile_templates: list[_EffectTemplate] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, reference_dir: str | Path):
        """
        Load all artifact and hero board effect reference images, plus any
        unit-folder tile images.

        Args:
            reference_dir: Root of the reference library (assets/reference/).
        """
        reference_dir = Path(reference_dir)
        self._templates.clear()
        self._empty_tile_templates.clear()

        self._load_subfolder(reference_dir / "artifacts",          "artifact")
        self._load_subfolder(reference_dir / "hero_board_effects", "hero_skill")
        self._load_unit_tiles(reference_dir)

        self._loaded = True
        n_art   = sum(1 for t in self._templates if t.effect_type == "artifact")
        n_hero  = sum(1 for t in self._templates if t.effect_type == "hero_skill")
        n_unit  = sum(1 for t in self._templates if t.effect_type == "unit_tile")
        n_empty = len(self._empty_tile_templates)
        print(f"[BoardEffectClassifier] Loaded {n_art} artifact + {n_hero} hero-skill"
              f" + {n_unit} unit-tile templates ({n_empty} empty-tile filters)"
              f" from {reference_dir}")

    def _load_subfolder(self, folder: Path, effect_type: str):
        """Load images from hero_board_effects/ or artifacts/ subfolders."""
        if not folder.exists():
            return
        for entity_dir in sorted(folder.iterdir()):
            if not entity_dir.is_dir():
                continue
            entity_id = entity_dir.name
            for img_path in sorted(entity_dir.rglob("*.png")):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                half = cv2.resize(img, (max(1, w // 2), max(1, h // 2)),
                                  interpolation=cv2.INTER_AREA)
                tmpl = _EffectTemplate(
                    effect_type=effect_type,
                    effect_id=entity_id,
                    effect_name=img_path.stem,
                    images=[img, half],
                )
                self._templates.append(tmpl)
                if _is_tile_stem(img_path.stem):
                    self._empty_tile_templates.append(tmpl)

    def _load_unit_tiles(self, reference_dir: Path):
        """Scan unit reference folders for tile/placed-object images.

        Two kinds of file are collected:
          1. Any file whose stem ends in _tile (e.g. cultist/sacrifice_tile-2.png)
          2. Any file with NO 'rank' token in its stem (e.g. kobold/kobold_mine.png,
             kobold/kobold_gold_mine.png) — these are skipped by TemplateMatcher's
             filename parser and represent placed objects, not fighting units.

        These are added ONLY to _empty_tile_templates (not to _templates) so they
        act purely as filters and never produce false classify() hits.
        """
        for unit_dir in sorted(reference_dir.iterdir()):
            if not unit_dir.is_dir() or unit_dir.name in _NON_UNIT_DIRS:
                continue
            unit_id = unit_dir.name
            for img_path in sorted(unit_dir.rglob("*.png")):
                stem = img_path.stem
                if not (_is_tile_stem(stem) or _is_rankless(stem)):
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                half = cv2.resize(img, (max(1, w // 2), max(1, h // 2)),
                                  interpolation=cv2.INTER_AREA)
                tmpl = _EffectTemplate(
                    effect_type="unit_tile",
                    effect_id=unit_id,
                    effect_name=stem,
                    images=[img, half],
                )
                self._templates.append(tmpl)
                self._empty_tile_templates.append(tmpl)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, cell_crop: np.ndarray) -> Optional[BoardEffect]:
        """
        Identify any artifact or hero skill effect visible in a cell crop.

        Returns BoardEffect if a match above threshold is found, else None.
        Returns None immediately if no templates are loaded or the crop is empty.
        """
        if not self._loaded or not self._templates or cell_crop.size == 0:
            return None

        best_score = -1.0
        best_tmpl: Optional[_EffectTemplate] = None

        for tmpl in self._templates:
            score = _best_ncc(cell_crop, tmpl.images)
            if score > best_score:
                best_score = score
                best_tmpl = tmpl

        if best_score < self.threshold or best_tmpl is None:
            return None

        return BoardEffect(
            effect_type=best_tmpl.effect_type,
            effect_id=best_tmpl.effect_id,
            effect_name=best_tmpl.effect_name,
            confidence=best_score,
        )

    # ------------------------------------------------------------------
    # Empty-tile filter
    # ------------------------------------------------------------------

    def is_empty_tile(self, cell_crop: np.ndarray,
                      threshold: float = 0.85) -> bool:
        """
        Return True if cell_crop matches a known empty-tile reference image.

        Empty tiles are reference images whose filename ends in '_tile'
        (e.g. wind_tile.png, relic_of_growth_tile.png). They represent the
        visual state of a board slot that has no unit — only a hero skill or
        artifact background effect. Crops matching these should NOT be saved
        to data/to_label/ since there is no unit present to label.

        Args:
            cell_crop: BGR crop of a single board cell.
            threshold: NCC score required to declare a match (default 0.85).
                       High value intentional — only suppress saves when the
                       match is nearly exact.
        """
        if not self._empty_tile_templates or cell_crop.size == 0:
            return False
        for tmpl in self._empty_tile_templates:
            if _best_ncc(cell_crop, tmpl.images) >= threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def loaded_ids(self) -> dict[str, list[str]]:
        """Returns {'artifacts': [...], 'hero_skills': [...]} of loaded IDs."""
        return {
            "artifacts":   sorted({t.effect_id for t in self._templates
                                   if t.effect_type == "artifact"}),
            "hero_skills": sorted({t.effect_id for t in self._templates
                                   if t.effect_type == "hero_skill"}),
        }

    def template_count(self) -> int:
        return len(self._templates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_ncc(query: np.ndarray, templates: list) -> float:
    """Best NCC score across a list of template scales."""
    best = -1.0
    for tmpl in templates:
        th, tw = tmpl.shape[:2]
        resized = cv2.resize(query, (tw, th), interpolation=cv2.INTER_AREA)
        result  = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
    return best