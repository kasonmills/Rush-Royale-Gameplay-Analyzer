"""
Phase 1 unit recognition via OpenCV template matching.

Rush Royale always displays the 5 units in each player's deck as persistent
icons beneath their board. This is the recognition anchor:

  Step 1 — Deck icon scan (once per match, or on first clear frame).
    The 5 deck icons below each player's board are matched against the full
    reference library to lock in both decks. This is the only time the full
    library is searched.

  Step 2 — Board cell matching (every frame, deck-constrained).
    Individual board cells are matched only against the ~5–10 reference images
    for the identified deck units. Fast and accurate since the search space
    is tiny.

This architecture means board cell recognition is always O(deck_size) rather
than O(full_library), and the deck lock-in is a one-time cost.

GridCalibrator is expected to provide both board cell crops AND deck icon crops.
The deck icon region should be added to calibration alongside the play boards.

Usage:
    matcher = TemplateMatcher()
    matcher.load_library("assets/reference")

    # Step 1: identify both decks from the deck icon strip
    player_deck  = matcher.identify_deck(deck_icon_crops_player)
    opponent_deck = matcher.identify_deck(deck_icon_crops_opponent)

    # Step 2: match board cells constrained to the identified decks
    result = matcher.match_cell(cell_crop, unit_ids=player_deck)
    print(result.unit_id, result.merge_rank, result.confidence)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Minimum NCC score to accept a board cell match.
CELL_CONFIDENCE_THRESHOLD = 0.60

# Minimum NCC score to accept a deck icon match (lower because icons are small).
DECK_ICON_THRESHOLD = 0.50

# Template heights used for matching (native and half-size for coarse-to-fine).
TEMPLATE_HEIGHTS = (96, 64)


@dataclass
class MatchResult:
    unit_id: Optional[str] = None
    merge_rank: Optional[int] = None
    appearance_state: str = "base"
    variant_tag: Optional[str] = None
    confidence: float = 0.0
    is_empty: bool = False


@dataclass
class _TemplateEntry:
    unit_id: str
    merge_rank: int
    appearance_state: str
    variant_tag: Optional[str]
    images: dict[int, np.ndarray] = field(default_factory=dict)


class TemplateMatcher:
    """
    Manages the visual reference library and performs:
      - Deck identification from the persistent deck icon strip.
      - Board cell matching constrained to the identified deck.

    Args:
        cell_threshold:      NCC score below which a board cell is unrecognised.
        deck_icon_threshold: NCC score below which a deck icon slot is unidentified.
    """

    def __init__(self,
                 cell_threshold: float = CELL_CONFIDENCE_THRESHOLD,
                 deck_icon_threshold: float = DECK_ICON_THRESHOLD):
        self.cell_threshold = cell_threshold
        self.deck_icon_threshold = deck_icon_threshold
        self._templates: list[_TemplateEntry] = []
        self._index: dict[str, list[_TemplateEntry]] = {}  # unit_id → entries
        self._loaded = False

    # ------------------------------------------------------------------
    # Library loading
    # ------------------------------------------------------------------

    def load_library(self, reference_dir: str | Path):
        """
        Loads reference images from the visual reference library.

        Expected directory layout (produced by tools/extract_assets.py):
          <reference_dir>/
            <unit_id>/
              <appearance_state>_rank<N>[_<variant>].png
              e.g.  base_rank1.png
                    max_level_rank7.png
                    reincarnation_1_rank7_moon.png

        Missing files are silently skipped.
        """
        reference_dir = Path(reference_dir)
        if not reference_dir.is_dir():
            raise FileNotFoundError(
                f"Reference directory not found: {reference_dir}"
            )

        self._templates.clear()
        self._index.clear()

        for unit_dir in sorted(reference_dir.iterdir()):
            if not unit_dir.is_dir():
                continue
            unit_id = unit_dir.name

            for img_path in sorted(unit_dir.glob("*.png")):
                entry = _parse_reference_filename(unit_id, img_path)
                if entry is None:
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                for h in TEMPLATE_HEIGHTS:
                    entry.images[h] = _resize_to_height(img, h)
                self._templates.append(entry)
                self._index.setdefault(unit_id, []).append(entry)

        self._loaded = True
        print(f"[TemplateMatcher] Loaded {len(self._templates)} templates "
              f"for {len(self._index)} units.")

    def loaded_unit_ids(self) -> set[str]:
        return set(self._index.keys())

    # ------------------------------------------------------------------
    # Step 1 — Deck identification
    # ------------------------------------------------------------------

    def identify_deck(self, icon_crops: list[np.ndarray]) -> set[str]:
        """
        Identify the 5 units in a player's deck from the persistent icon strip.

        Args:
            icon_crops: List of 5 cropped images, one per deck slot. Obtained
                        from GridCalibrator.crop_deck_icons().

        Returns:
            Set of unit_id strings for the identified deck. Slots that fall
            below the deck icon threshold are omitted; the caller should retry
            on the next clear frame if the set has fewer than 5 entries.
        """
        if not self._loaded:
            raise RuntimeError("load_library() must be called first.")

        identified: set[str] = set()
        for crop in icon_crops:
            result = self._match_against_full_library(crop,
                                                      self.deck_icon_threshold)
            if result.unit_id is not None:
                identified.add(result.unit_id)
        return identified

    # ------------------------------------------------------------------
    # Step 2 — Board cell matching (deck-constrained)
    # ------------------------------------------------------------------

    def match_cell(self,
                   cell_crop: np.ndarray,
                   unit_ids: Optional[set[str]] = None) -> MatchResult:
        """
        Match a board cell crop against the reference library.

        Args:
            cell_crop: BGR image of a single board cell.
            unit_ids:  Deck-constrained set of unit IDs to search. Pass the
                       set returned by identify_deck(). If None, falls back to
                       searching the full library (slower; use for diagnostics
                       or when deck is not yet identified).

        Returns:
            MatchResult with best match, or is_empty=True if the cell is dark/blank,
            or confidence=0 if no match exceeds the threshold.
        """
        if not self._loaded:
            raise RuntimeError("load_library() must be called first.")

        if _is_empty_cell(cell_crop):
            return MatchResult(is_empty=True)

        templates = (
            self._templates if unit_ids is None
            else [t for uid in unit_ids for t in self._index.get(uid, [])]
        )
        if not templates:
            return MatchResult()

        best_score = -1.0
        best_entry: Optional[_TemplateEntry] = None

        for entry in templates:
            score = _max_ncc_score(cell_crop, entry)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score < self.cell_threshold or best_entry is None:
            return MatchResult(confidence=best_score)

        return MatchResult(
            unit_id=best_entry.unit_id,
            merge_rank=best_entry.merge_rank,
            appearance_state=best_entry.appearance_state,
            variant_tag=best_entry.variant_tag,
            confidence=best_score,
        )

    def match_all_cells(self,
                        cell_crops: list[tuple[str, int, int, np.ndarray]],
                        player_deck: Optional[set[str]] = None,
                        opponent_deck: Optional[set[str]] = None
                        ) -> list[tuple[str, int, int, MatchResult]]:
        """
        Batch-match all cells from GridCalibrator.all_cell_crops().

        Uses player_deck for player cells and opponent_deck for opponent cells,
        so each side is constrained to its own identified units.

        Returns list of (player, row, col, MatchResult).
        """
        results = []
        for player, row, col, crop in cell_crops:
            deck = player_deck if player == "player" else opponent_deck
            results.append((player, row, col, self.match_cell(crop, deck)))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_against_full_library(self,
                                    crop: np.ndarray,
                                    threshold: float) -> MatchResult:
        """Full-library search used only for deck icon identification."""
        if _is_empty_cell(crop):
            return MatchResult(is_empty=True)
        best_score = -1.0
        best_entry: Optional[_TemplateEntry] = None
        for entry in self._templates:
            score = _max_ncc_score(crop, entry)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score < threshold or best_entry is None:
            return MatchResult(confidence=best_score)
        return MatchResult(
            unit_id=best_entry.unit_id,
            merge_rank=best_entry.merge_rank,
            appearance_state=best_entry.appearance_state,
            variant_tag=best_entry.variant_tag,
            confidence=best_score,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_reference_filename(unit_id: str,
                               path: Path) -> Optional[_TemplateEntry]:
    """
    Parses a reference image filename.
    Format: <appearance_state>_rank<N>[_<variant>].png
    """
    stem = path.stem
    parts = stem.split("_")
    rank_idx = next(
        (i for i, p in enumerate(parts) if p.startswith("rank") and p[4:].isdigit()),
        None
    )
    if rank_idx is None:
        return None
    merge_rank = int(parts[rank_idx][4:])
    appearance_state = "_".join(parts[:rank_idx]) or "base"
    variant_tag = "_".join(parts[rank_idx + 1:]) or None
    return _TemplateEntry(
        unit_id=unit_id,
        merge_rank=merge_rank,
        appearance_state=appearance_state,
        variant_tag=variant_tag,
    )


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (max(1, int(w * scale)), target_h),
                      interpolation=cv2.INTER_AREA)


def _max_ncc_score(cell_crop: np.ndarray, entry: _TemplateEntry) -> float:
    """Peak NCC score across all pre-scaled templates for this entry."""
    best = -1.0
    for target_h, template in entry.images.items():
        resized = _resize_to_height(cell_crop, target_h)
        th, tw = template.shape[:2]
        ch, cw = resized.shape[:2]
        if ch < th or cw < tw:
            continue
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
    return best


def _is_empty_cell(cell_crop: np.ndarray) -> bool:
    """Returns True if the cell appears to be empty (dark or uniform)."""
    if cell_crop.size == 0:
        return True
    gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
    mean_val, std_dev = cv2.meanStdDev(gray)
    return float(mean_val[0][0]) < 15 or float(std_dev[0][0]) < 8
