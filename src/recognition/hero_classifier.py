"""
Hero portrait classifier — identifies which hero each player has equipped
by matching the HUD portrait region against reference images.

Rush Royale PvP HUD layout (portrait orientation):
  - Both players' hero portraits are visible throughout the match.
  - Player portrait: lower HUD area, to the left of the player's HP bar.
  - Opponent portrait: upper HUD area, to the left of the opponent's HP bar.

The portrait is a small icon (roughly square). Default region fractions
are tuned for a 1080×2340 scrcpy portrait stream and can be overridden
by passing a custom HUDRegions instance.

Reference images live in:
  assets/reference/hero_portraits/
    <hero_id>.png   (e.g.  mermaid.png, zeus.png, necromancer.png)

  One image per hero. The filename stem must match the hero_id stored in
  the heroes table of unit_meta.db.

Usage:
    classifier = HeroClassifier()
    classifier.load("assets/reference/hero_portraits")

    player_crop   = classifier.crop_portrait(frame, "player")
    opponent_crop = classifier.crop_portrait(frame, "opponent")

    result = classifier.classify(player_crop)
    if result:
        print(result.hero_id, result.confidence)

    # Or classify both sides at once:
    ids = classifier.classify_frame(frame)
    # ids == {"player": HeroResult(...), "opponent": HeroResult(...)}
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# HUD portrait region — fractions of (frame_width, frame_height)
# Tuned for a 1080×2340 portrait stream.
# Each region is (left_frac, top_frac, right_frac, bottom_frac).
# ---------------------------------------------------------------------------

# Player portrait: lower area of the frame, left of the HP bar
_PLAYER_PORTRAIT = (0.02, 0.91, 0.14, 0.99)

# Opponent portrait: upper area of the frame, left of the HP bar
_OPP_PORTRAIT    = (0.02, 0.01, 0.14, 0.09)


@dataclass
class HUDRegions:
    """
    Portrait crop regions as frame fractions (left, top, right, bottom).
    Override to adapt to a different stream resolution or UI layout.
    """
    player_portrait:   tuple[float, float, float, float] = _PLAYER_PORTRAIT
    opponent_portrait: tuple[float, float, float, float] = _OPP_PORTRAIT


@dataclass
class HeroResult:
    """
    The hero identified from a portrait crop.

    hero_id:    Matches heroes.hero_id in unit_meta.db.
    confidence: NCC match score (0.0–1.0).
    """
    hero_id: str
    confidence: float


class HeroClassifier:
    """
    Matches HUD hero portrait crops against reference images.

    Args:
        threshold: NCC score below which no hero is reported.
        regions:   HUD crop fractions. Use default or pass a custom HUDRegions.
    """

    def __init__(self,
                 threshold: float = CONFIDENCE_THRESHOLD,
                 regions: HUDRegions = HUDRegions()):
        self.threshold = threshold
        self.regions = regions
        # hero_id → [full-size img, half-size img]
        self._templates: dict[str, list[np.ndarray]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, portrait_dir: str | Path):
        """
        Load hero portrait reference images from portrait_dir.

        Expected filenames: <hero_id>.png  (e.g. mermaid.png)
        Missing or unreadable files are silently skipped.
        """
        portrait_dir = Path(portrait_dir)
        self._templates.clear()

        for path in sorted(portrait_dir.glob("*.png")):
            hero_id = path.stem
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            self._templates[hero_id] = [
                img,
                cv2.resize(img, (max(1, w // 2), max(1, h // 2)),
                           interpolation=cv2.INTER_AREA),
            ]

        self._loaded = True
        print(f"[HeroClassifier] Loaded {len(self._templates)} hero portrait "
              f"templates from {portrait_dir}")

    def loaded_hero_ids(self) -> set[str]:
        return set(self._templates.keys())

    # ------------------------------------------------------------------
    # Region cropping
    # ------------------------------------------------------------------

    def crop_portrait(self, frame: np.ndarray, player: str) -> np.ndarray:
        """
        Crop the hero portrait region from a full frame.

        Args:
            frame:  Full BGR frame from the capture pipeline.
            player: 'player' or 'opponent'.

        Returns:
            Cropped BGR portrait image. May be empty (0×0) if the region
            falls outside the frame — callers should check .size > 0.
        """
        fh, fw = frame.shape[:2]
        region = (self.regions.player_portrait if player == "player"
                  else self.regions.opponent_portrait)
        l, t, r, b = region
        x1 = max(0, int(fw * l))
        y1 = max(0, int(fh * t))
        x2 = min(fw, int(fw * r))
        y2 = min(fh, int(fh * b))
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, portrait_crop: np.ndarray) -> Optional[HeroResult]:
        """
        Identify the hero shown in a portrait crop.

        Returns HeroResult if a match above the confidence threshold is found,
        or None if no hero can be identified (e.g. no templates loaded, crop
        too small, or best score below threshold).
        """
        if not self._loaded or portrait_crop.size == 0:
            return None

        best_score = -1.0
        best_hero: Optional[str] = None

        for hero_id, scaled_imgs in self._templates.items():
            score = _best_ncc(portrait_crop, scaled_imgs)
            if score > best_score:
                best_score = score
                best_hero = hero_id

        if best_score < self.threshold or best_hero is None:
            return None

        return HeroResult(hero_id=best_hero, confidence=best_score)

    def classify_frame(self, frame: np.ndarray
                       ) -> dict[str, Optional[HeroResult]]:
        """
        Classify both hero portraits from a full frame in one call.

        Returns:
            {"player": HeroResult | None, "opponent": HeroResult | None}
        """
        return {
            "player":   self.classify(self.crop_portrait(frame, "player")),
            "opponent": self.classify(self.crop_portrait(frame, "opponent")),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
