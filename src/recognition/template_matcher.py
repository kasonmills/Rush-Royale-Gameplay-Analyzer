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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.recognition.shape_detector import detect_merge_rank


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
    is_toplevel: bool = True   # False when loaded from a subfolder
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
                 deck_icon_threshold: float = DECK_ICON_THRESHOLD,
                 use_rotation: bool = False,
                 use_shape_mask: bool = False):
        self.cell_threshold = cell_threshold
        self.deck_icon_threshold = deck_icon_threshold
        self.use_rotation = use_rotation
        self.use_shape_mask = use_shape_mask
        self._templates: list[_TemplateEntry] = []
        self._index: dict[str, list[_TemplateEntry]] = {}  # unit_id → entries
        self._loaded = False

    # ------------------------------------------------------------------
    # Library loading
    # ------------------------------------------------------------------

    def load_library(self, reference_dir: str | Path):
        """
        Loads reference images from the visual reference library.

        Directory layout — both flat and subfolder structures are supported:

          Flat (existing):
            <unit_id>/
              <appearance_state>_rank<N>[_<variant>].png
              e.g.  base_rank1.png  max_rank7.png  active_rank_1.png

          Subfolder (branch-organised):
            <unit_id>/
              rank<N>.png              ← toplevel: base / most-common state
              <branch>/
                rank<N>[_<variant>].png

          When a file is inside a subfolder, the folder name is prepended as
          the appearance-state prefix automatically.  So:
            cultist/active/rank_1.png  ≡  cultist/active_rank_1.png

          Matching is two-phase: toplevel files are checked first; subfolders
          are searched only when the toplevel pass does not produce a confident
          match (>= cell_threshold).  This prunes the search space for the
          common case where a unit is in its base state.

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

            for img_path in sorted(unit_dir.rglob("*.png")):
                rel        = img_path.relative_to(unit_dir)
                is_toplevel = rel.parent == Path(".")
                subdir     = "" if is_toplevel else (
                    str(rel.parent).replace("\\", "_").replace("/", "_")
                )
                entry = _parse_reference_filename(unit_id, img_path, subdir)
                if entry is None:
                    continue
                entry.is_toplevel = is_toplevel
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

        # Detect the merge-rank shape in the live crop and generate a mask.
        # When use_shape_mask=False both values are unused (no overhead).
        if self.use_shape_mask:
            detected_rank, shape_mask = detect_merge_rank(cell_crop)
        else:
            detected_rank, shape_mask = 0, None

        deck_constrained = unit_ids is not None and len(unit_ids) > 0

        if deck_constrained:
            # Deck-constrained two-phase search (used by the analysis pipeline).
            # Toplevel (base-state) images are checked first; subfolders are
            # only searched when the toplevel pass is not confident.  This
            # prunes the search space — the deck is known so the library is
            # already small.
            all_templates = [t for uid in unit_ids
                             for t in self._index.get(uid, [])]
            if not all_templates:
                return MatchResult()

            toplevel = [t for t in all_templates if t.is_toplevel]
            best_score, best_entry = self._best_match(
                cell_crop, toplevel, detected_rank, shape_mask)

            if best_score < self.cell_threshold:
                subdir_entries = [t for t in all_templates if not t.is_toplevel]
                if subdir_entries:
                    sub_score, sub_entry = self._best_match(
                        cell_crop, subdir_entries, detected_rank, shape_mask)
                    if sub_score > best_score:
                        best_score, best_entry = sub_score, sub_entry
        else:
            # Full-library flat search — no deck constraint (used by
            # extract_cells and diagnostics).  All templates searched at once
            # to maximise recall; subfolder organisation is transparent.
            if not self._templates:
                return MatchResult()
            best_score, best_entry = self._best_match(
                cell_crop, self._templates, detected_rank, shape_mask)

        if best_score < self.cell_threshold or best_entry is None:
            return MatchResult(confidence=best_score)

        return MatchResult(
            unit_id=best_entry.unit_id,
            merge_rank=best_entry.merge_rank,
            appearance_state=best_entry.appearance_state,
            variant_tag=best_entry.variant_tag,
            confidence=best_score,
        )

    def _best_match(self,
                    cell_crop: np.ndarray,
                    entries: list[_TemplateEntry],
                    detected_rank: int = 0,
                    shape_mask: Optional[np.ndarray] = None,
                    ) -> tuple[float, Optional[_TemplateEntry]]:
        """Return (best_score, best_entry) for a list of template entries.

        When detected_rank > 0 and shape_mask is provided, entries whose
        merge_rank does not match detected_rank are skipped (rank-constrained
        search), and masked NCC is used for the remaining comparisons.
        If ALL entries are filtered out by the rank check, the rank filter is
        relaxed so that at least the unmasked best match is returned.
        """
        best_score = -1.0
        best_entry: Optional[_TemplateEntry] = None

        rank_filter_active = (detected_rank > 0 and shape_mask is not None)

        for entry in entries:
            if rank_filter_active and entry.merge_rank != detected_rank:
                continue
            score = _max_ncc_score(cell_crop, entry, self.use_rotation,
                                   shape_mask)
            if score > best_score:
                best_score = score
                best_entry = entry

        # Fallback: if rank filtering eliminated every candidate, retry without
        # the rank constraint (still apply the mask if available).
        if best_entry is None and rank_filter_active:
            for entry in entries:
                score = _max_ncc_score(cell_crop, entry, self.use_rotation,
                                       shape_mask)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        return best_score, best_entry

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
                               path: Path,
                               subdir_prefix: str = "") -> Optional[_TemplateEntry]:
    """
    Parses a reference image filename.

    Flat format:
      <appearance_state>_rank<N>[_<variant>].png   e.g. active_rank1.png
      <appearance_state>_rank_<N>[_<variant>].png  e.g. cordyceps_max_rank_7.png

    Subfolder format (subdir_prefix is the folder name, prepended automatically):
      <branch>/rank<N>.png  →  effective stem  <branch>_rank<N>

    Everything before the rank token is the appearance state; everything after
    is the variant tag.
    """
    stem  = f"{subdir_prefix}_{path.stem}" if subdir_prefix else path.stem
    parts = stem.split("_")

    # Find the index of the "rank" token. Accepts both rank1 (digit attached)
    # and rank + separate digit token (rank_1 style).
    rank_idx   = None
    merge_rank = None
    skip       = 0   # extra tokens consumed when rank and digit are separate

    for i, p in enumerate(parts):
        if not p.startswith("rank"):
            continue
        after = p[4:]
        # Strip a trailing -N variant suffix (e.g. "rank1-2" → "1", "rank7-3" → "7")
        rank_str = after.split("-")[0]
        if rank_str.isdigit():
            # Compact form: rank7  or  rank1-2
            rank_idx   = i
            merge_rank = int(rank_str)
            break
        if after == "" and i + 1 < len(parts):
            # Spaced form: rank_7  or  rank_1-2  (two tokens)
            next_rank_str = parts[i + 1].split("-")[0]
            if next_rank_str.isdigit():
                rank_idx   = i
                merge_rank = int(next_rank_str)
                skip       = 1
                break

    if rank_idx is None:
        return None

    appearance_state = "_".join(parts[:rank_idx]) or "base"
    variant_tag      = "_".join(parts[rank_idx + 1 + skip:]) or None
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


# Rotation angles tried when use_rotation=True.  0° is always tried first;
# these are the additional angles.  ±15° and ±30° cover the typical unit
# orientation variance in Rush Royale without excessive compute.
_EXTRA_ANGLES = (15, -15, 30, -30)


def _rotate_crop(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _masked_ncc(crop_gray: np.ndarray,
                tmpl_gray: np.ndarray,
                mask: np.ndarray) -> float:
    """
    Pearson NCC computed only over pixels where mask > 0.

    This is mathematically correct masked NCC: the mean and standard deviation
    are calculated from the masked pixels only, so zeroed-out background pixels
    don't distort the result (unlike passing zeroed images to matchTemplate).

    Returns a value in [-1, 1], or -1.0 when the mask covers too few pixels.
    Both crop_gray and tmpl_gray must already be the same (h, w).
    """
    idx = mask > 0
    n = int(idx.sum())
    if n < 64:
        return -1.0

    a = crop_gray[idx].astype(np.float64)
    b = tmpl_gray[idx].astype(np.float64)

    a -= a.mean()
    b -= b.mean()

    denom = math.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-6:
        return 0.0

    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _max_ncc_score(cell_crop: np.ndarray,
                   entry: _TemplateEntry,
                   use_rotation: bool = False,
                   shape_mask: Optional[np.ndarray] = None) -> float:
    """Peak NCC score across all pre-scaled templates for this entry.

    When use_rotation=True, also tries ±15° and ±30° rotations of the crop
    and returns the best score found.  This accounts for unit sprites that
    face different directions depending on which enemy they are targeting.
    Rotation is opt-in because it multiplies matching cost by ~5x.

    When shape_mask is provided (a binary uint8 image from shape_detector),
    masked NCC is used instead of standard matchTemplate.  The mask is resized
    to match each template's dimensions before comparison.  This eliminates
    background/grid pixels from the correlation, giving cleaner scores.
    """
    best = -1.0
    crops_to_try = [cell_crop]
    if use_rotation:
        crops_to_try += [_rotate_crop(cell_crop, a) for a in _EXTRA_ANGLES]

    use_mask = shape_mask is not None

    for crop in crops_to_try:
        # Convert to grayscale once per crop variant when masking is active
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if use_mask else None

        for _, template in entry.images.items():
            th, tw = template.shape[:2]

            if use_mask:
                crop_resized = cv2.resize(crop_gray, (tw, th),
                                          interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(shape_mask, (tw, th),
                                          interpolation=cv2.INTER_NEAREST)
                tmpl_gray    = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                score        = _masked_ncc(crop_resized, tmpl_gray, mask_resized)
            else:
                resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
                result  = cv2.matchTemplate(resized, template,
                                            cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)

            if score > best:
                best = score
    return best


def _is_empty_cell(cell_crop: np.ndarray) -> bool:
    """Returns True if the cell appears to be empty (dark AND uniform)."""
    if cell_crop.size == 0:
        return True
    gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
    mean_val, std_dev = cv2.meanStdDev(gray)
    return float(mean_val[0][0]) < 15 and float(std_dev[0][0]) < 8
