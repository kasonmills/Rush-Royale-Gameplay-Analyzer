"""
Match Context Resolver (MCR) — orchestrates the full recognition pipeline
across the lifetime of a single match session.

The MCR binds together all Phase 1 recognition modules and adds the
cross-frame state accumulation needed for reliable in-match tracking.

Six stages run each time process_frame() is called:

  Stage 1 — Deck identification (once, retried until successful)
    Locks in both 5-unit decks from the persistent deck icon strip.
    Scans every frame until both decks reach the expected size, then
    switches all subsequent board matching to deck-constrained mode.
    If decks cannot be identified after _MAX_DECK_SCAN_ATTEMPTS frames,
    the best partial result is locked in and a warning is printed.

  Stage 2 — Board cell matching (every frame, deck-constrained)
    All 30 cells are matched via TemplateMatcher, constrained to the
    identified decks.  Before decks are locked the full library is used.

  Stage 3 — Talent path accumulation (every frame)
    TalentClassifier detects the highest badge on each occupied cell.
    Lower tiers are resolved by:
      a) Merging observations from earlier frames (cached per unit).
      b) DB lookup for tiers known to be Fixed for this unit.
    Branching tiers that have never been directly observed remain None.

  Stage 4 — Hero identification (first frame, then periodic refresh)
    HeroClassifier runs on the portrait regions and the result is cached.
    Runs again every _HERO_SCAN_INTERVAL frames to catch late-join VODs.

  Stage 5 — HUD reading (every frame)
    OCRReader reads wave number, player/opponent HP, and player mana.

  Stage 6 — GameState assembly
    All outputs are merged into a single GameState snapshot.  The
    pipeline_confidence field is the mean recognition_confidence across
    all occupied cells.

Special-case handling embedded in Stage 2/3:
  - Treant: merge_rank capped at 4 (checked against DB max_merge_rank
    when a connection is available; falls back to a hardcoded set).
  - Twins: moon/sun variant_tag is passed through from MatchResult
    with no extra processing needed.
  - Other Phase 3 special cases (Inquisitor stacks, Enchanted Sword
    colour, Cultist transformation, etc.) are noted but deferred.

Usage:
    matcher       = TemplateMatcher(); matcher.load_library("assets/reference")
    talent_clf    = TalentClassifier(); talent_clf.load("assets/reference/talent_icons")
    hero_clf      = HeroClassifier();  hero_clf.load("assets/reference/hero_portraits")
    ocr_reader    = OCRReader()
    calibrator    = GridCalibrator.from_defaults(1080, 2340)

    mcr     = MatchContextResolver(matcher, talent_clf, hero_clf, ocr_reader)
    session = mcr.start_match(match_id="abc123")

    for frame, timestamp in video_source:
        # db_conn is optional; enables Fixed-tier DB lookup and rank-cap
        with unit_meta_db() as db_conn:
            state = mcr.process_frame(frame, calibrator, session,
                                      timestamp, db_conn=db_conn)
        # state is a fully populated GameState ready for the predictor or DB
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.analysis.game_state import GameState, UnitCell
from src.capture.grid_calibrator import GridCalibrator
from src.recognition.animation_detector import AnimationDetector
from src.recognition.hero_classifier import HeroClassifier
from src.recognition.ocr_reader import OCRReader
from src.recognition.rank_detector import RankDetector
from src.recognition.talent_classifier import TalentClassifier
from src.recognition.template_matcher import MatchResult, TemplateMatcher
from src.recognition.unit_classifier import UnitClassifier


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Give up waiting for a full deck identification after this many frames.
# Partial decks will be locked and a warning printed.
_MAX_DECK_SCAN_ATTEMPTS = 60

# Expected units per deck (standard Rush Royale deck size).
_DECK_SIZE = 5

# Re-run hero classification every N frames (portraits are stable; this is
# just a safeguard for VODs where the player isn't on screen yet at frame 0).
_HERO_SCAN_INTERVAL = 150

# Treant hard-capped at rank 4.  Overridden by DB max_merge_rank if available.
_HARDCODED_RANK_CAPS: dict[str, int] = {
    "treant": 4,
}

# pipeline_confidence below this value on a frame is flagged as low quality.
_LOW_CONFIDENCE_WARN = 0.45


# ---------------------------------------------------------------------------
# MatchSession — mutable state accumulated across frames
# ---------------------------------------------------------------------------

@dataclass
class MatchSession:
    """
    All cross-frame state for one active match.

    Create with MatchContextResolver.start_match() rather than directly.
    """

    match_id: str
    started_at: float = field(default_factory=time.time)
    frame_count: int = 0

    # ----- deck identification -----------------------------------------
    player_deck: set[str] = field(default_factory=set)
    opponent_deck: set[str] = field(default_factory=set)
    decks_locked: bool = False
    deck_scan_attempts: int = 0

    # ----- hero identities (cached after first clear detection) ---------
    player_hero_id: Optional[str] = None
    opponent_hero_id: Optional[str] = None

    # ----- talent observations -----------------------------------------
    # (player, unit_id) → {tier: branch}
    # Populated incrementally: each frame adds the observed tier/branch.
    # Fixed tiers fetched from DB are also stored here to avoid repeated lookups.
    talent_cache: dict[tuple[str, str], dict[int, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    # ----- rank cap cache (from DB, populated on first DB-backed frame) -
    # unit_id → max_merge_rank  (None = not yet looked up)
    _rank_cap_cache: dict[str, Optional[int]] = field(
        default_factory=dict
    )

    # ----- last assembled state ----------------------------------------
    last_state: Optional[GameState] = None


# ---------------------------------------------------------------------------
# MatchContextResolver
# ---------------------------------------------------------------------------

class MatchContextResolver:
    """
    Processes frames for an active match and returns a populated GameState.

    Args:
        matcher:            Loaded TemplateMatcher.
        talent_classifier:  Loaded TalentClassifier.
        hero_classifier:    Loaded HeroClassifier.
        ocr_reader:         OCRReader instance.
        animation_detector: Optional AnimationDetector.  When provided,
                            Stage 3.5 runs after board assembly and populates
                            player_active_buffs / opponent_active_buffs on
                            each GameState.  When None, those dicts remain empty.
        unit_classifier:    Optional UnitClassifier.  When provided and loaded,
                            its top-1 prediction overrides the template matcher's
                            unit_id for any cell where the classifier confidence
                            exceeds the NCC score.  Rank and appearance_state
                            are always taken from the template matcher.
    """

    def __init__(self,
                 matcher: TemplateMatcher,
                 talent_classifier: TalentClassifier,
                 hero_classifier: HeroClassifier,
                 ocr_reader: OCRReader,
                 animation_detector: Optional[AnimationDetector] = None,
                 unit_classifier: Optional[UnitClassifier] = None,
                 rank_detector: Optional[RankDetector] = None):
        self._matcher    = matcher
        self._talent     = talent_classifier
        self._hero       = hero_classifier
        self._ocr        = ocr_reader
        self._animations = animation_detector
        self._classifier = unit_classifier
        self._rank_det   = rank_detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_match(self, match_id: str,
                    started_at: Optional[float] = None) -> MatchSession:
        """
        Initialise a new MatchSession for a match.  Call once per match
        before the first process_frame() call.
        """
        return MatchSession(
            match_id=match_id,
            started_at=started_at if started_at is not None else time.time(),
        )

    def process_frame(self,
                      frame,
                      calibrator: GridCalibrator,
                      session: MatchSession,
                      timestamp_sec: float,
                      db_conn=None) -> GameState:
        """
        Run the full recognition pipeline on one frame and return a GameState.

        Args:
            frame:         BGR ndarray from VideoCapture / WindowCapture.
            calibrator:    Calibrated GridCalibrator for this stream.
            session:       MatchSession to read from and update.
            timestamp_sec: Position of this frame within the match / video.
            db_conn:       Optional sqlite3 connection to unit_meta.db.
                           Enables Fixed-tier DB lookup and DB-sourced rank caps.
                           If None, the MCR relies solely on cross-frame
                           observation for talent paths and uses hard-coded
                           rank caps only.

        Returns:
            Fully populated GameState snapshot.
        """
        session.frame_count += 1

        state = GameState(
            timestamp_sec=timestamp_sec,
            match_id=session.match_id,
            source_frame_index=session.frame_count,
        )

        # Stage 1 — deck identification (retried until locked)
        if not session.decks_locked:
            self._try_identify_decks(frame, calibrator, session)

        # Stage 2 + 3 — board cells + talent path accumulation
        self._process_boards(frame, calibrator, session, state, db_conn)

        # Stage 3.5 — animation detection (optional, when detector is wired in)
        if self._animations is not None:
            self._process_animations(frame, calibrator, state)

        # Stage 4 — hero identification (cached, periodic refresh)
        self._process_heroes(frame, session, state)

        # Stage 5 — HUD OCR
        self._process_hud(frame, state)

        # Stage 6 — pipeline confidence
        all_cells: list[UnitCell] = (
            [cell for _, _, cell in state.player_board.occupied()] +
            [cell for _, _, cell in state.opponent_board.occupied()]
        )
        if all_cells:
            state.pipeline_confidence = (
                sum(c.recognition_confidence for c in all_cells) / len(all_cells)
            )
            if state.pipeline_confidence < _LOW_CONFIDENCE_WARN:
                print(f"[MCR] frame {session.frame_count}: low pipeline "
                      f"confidence {state.pipeline_confidence:.2f}")

        session.last_state = state
        return state

    # ------------------------------------------------------------------
    # Stage 1 — Deck identification
    # ------------------------------------------------------------------

    def _try_identify_decks(self,
                             frame,
                             calibrator: GridCalibrator,
                             session: MatchSession) -> None:
        """
        Try to identify both decks from the deck icon strip.

        Accumulates hits across retries: if a unit appears in multiple
        scan attempts it is more likely to be a true deck member.
        Locks when both sides reach _DECK_SIZE.
        After _MAX_DECK_SCAN_ATTEMPTS the best partial result is locked.
        """
        session.deck_scan_attempts += 1

        player_icons = calibrator.crop_deck_icons(frame, "player")
        opp_icons    = calibrator.crop_deck_icons(frame, "opponent")

        # Accumulate — don't replace, so noisy frames don't clobber good ones
        session.player_deck.update(
            self._matcher.identify_deck(player_icons)
        )
        session.opponent_deck.update(
            self._matcher.identify_deck(opp_icons)
        )

        player_ready = len(session.player_deck) >= _DECK_SIZE
        opp_ready    = len(session.opponent_deck) >= _DECK_SIZE

        if player_ready and opp_ready:
            session.decks_locked = True
            print(f"[MCR] Decks locked (frame {session.frame_count}): "
                  f"player={session.player_deck}  "
                  f"opponent={session.opponent_deck}")
            return

        if session.deck_scan_attempts >= _MAX_DECK_SCAN_ATTEMPTS:
            session.decks_locked = True
            print(
                f"[MCR] WARNING — deck identification timed out after "
                f"{session.deck_scan_attempts} attempts. "
                f"player={session.player_deck} ({len(session.player_deck)}/5)  "
                f"opponent={session.opponent_deck} ({len(session.opponent_deck)}/5). "
                f"Proceeding with partial deck — cell matching accuracy may be reduced."
            )

    # ------------------------------------------------------------------
    # Stage 2 + 3 — Board cells and talent paths
    # ------------------------------------------------------------------

    def _process_boards(self,
                         frame,
                         calibrator: GridCalibrator,
                         session: MatchSession,
                         state: GameState,
                         db_conn) -> None:
        """Match all board cells and classify talent badges."""
        all_crops = calibrator.all_cell_crops(frame)

        # Use deck constraints only after decks are locked
        player_deck  = session.player_deck  if session.decks_locked else None
        opp_deck     = session.opponent_deck if session.decks_locked else None

        cell_results   = self._matcher.match_all_cells(all_crops, player_deck, opp_deck)
        talent_results = self._talent.classify_all(all_crops)

        # Stage 2.5 — classifier enhancement (when UnitClassifier is loaded).
        # For each cell where the classifier's confidence exceeds the NCC score,
        # override unit_id with the classifier's prediction.  Rank and
        # appearance_state are always preserved from the template matcher.
        if self._classifier is not None and self._classifier.available:
            crops_by_key = {(p, r, c): crop for p, r, c, crop in all_crops}
            for i, (player, row, col, match) in enumerate(cell_results):
                if match.is_empty or match.unit_id is None:
                    continue
                crop = crops_by_key.get((player, row, col))
                if crop is None:
                    continue
                deck = player_deck if player == "player" else opp_deck
                clf = self._classifier.classify(crop, deck)
                if clf is not None and clf.confidence > match.confidence:
                    cell_results[i] = (player, row, col, MatchResult(
                        unit_id=clf.unit_id,
                        merge_rank=match.merge_rank,
                        appearance_state=match.appearance_state,
                        variant_tag=match.variant_tag,
                        confidence=clf.confidence,
                    ))

        crops_by_key = {(p, r, c): crop for p, r, c, crop in all_crops}

        for player, row, col, match in cell_results:
            if match.is_empty or match.unit_id is None:
                continue

            # Shape-based rank detection — overrides template rank when available.
            # The border polygon is independent of unit artwork, so this is
            # reliable as soon as the cell is clearly cropped.
            rank_from_template = match.merge_rank or 1
            if self._rank_det is not None:
                crop = crops_by_key.get((player, row, col))
                detected_rank = self._rank_det.detect(crop) if crop is not None else None
                rank_from_template = detected_rank if detected_rank is not None else rank_from_template

            # Stage 2 special case — rank cap enforcement
            rank = self._apply_rank_cap(
                match.unit_id, rank_from_template, session, db_conn
            )

            # Stage 3 — talent path for this cell
            talent_result = talent_results.get((player, row, col))
            talent_path   = self._build_talent_path(
                player, match.unit_id, talent_result, session, db_conn
            )

            cell = UnitCell(
                unit_id=match.unit_id,
                merge_rank=rank,
                talent_path=talent_path,
                appearance_state=match.appearance_state,
                variant_tag=match.variant_tag,
                recognition_confidence=match.confidence,
            )

            board = (state.player_board if player == "player"
                     else state.opponent_board)
            board.set(row, col, cell)

    # ------------------------------------------------------------------
    # Stage 3 helpers — talent path accumulation
    # ------------------------------------------------------------------

    def _build_talent_path(self,
                            player: str,
                            unit_id: str,
                            talent_result,
                            session: MatchSession,
                            db_conn) -> dict[int, Optional[str]]:
        """
        Build the full cumulative talent_path for one cell.

        If no talent badge is visible this frame, returns an empty dict.
        Otherwise resolves all tiers 1..observed using:
          1. The observed branch for the highest badge (from TalentClassifier).
          2. Previously cached observations for lower tiers.
          3. DB lookup for Fixed lower tiers (requires db_conn).
          4. None for branching lower tiers that have never been observed.
        """
        if talent_result is None:
            return {}

        key   = (player, unit_id)
        cache = session.talent_cache[key]

        # Record this frame's observation
        cache[talent_result.tier] = talent_result.branch

        path: dict[int, Optional[str]] = {}
        for tier in range(1, talent_result.tier + 1):
            if tier == talent_result.tier:
                path[tier] = talent_result.branch

            elif tier in cache:
                # Seen in a previous frame
                path[tier] = cache[tier]

            elif db_conn is not None:
                # Ask the DB whether this is a Fixed tier for this unit
                branch = _db_lookup_fixed_branch(db_conn, unit_id, tier)
                if branch is not None:
                    path[tier] = branch
                    cache[tier] = branch   # cache so we don't query every frame
                else:
                    path[tier] = None      # branching, not yet observed
            else:
                path[tier] = None

        return path

    # ------------------------------------------------------------------
    # Stage 2 helper — rank cap
    # ------------------------------------------------------------------

    def _apply_rank_cap(self,
                         unit_id: str,
                         rank: int,
                         session: MatchSession,
                         db_conn) -> int:
        """
        Return the rank capped at the unit's maximum merge rank.

        Checks (in order):
          1. Session cache (avoids repeated DB queries per unit).
          2. DB max_merge_rank (if db_conn available).
          3. Hard-coded fallback table (_HARDCODED_RANK_CAPS).
          4. Default cap of 7 (standard Rush Royale maximum).
        """
        if unit_id not in session._rank_cap_cache:
            cap: Optional[int] = None

            if db_conn is not None:
                row = db_conn.execute(
                    "SELECT max_merge_rank FROM units WHERE unit_id = ?",
                    (unit_id,)
                ).fetchone()
                if row is not None:
                    cap = row["max_merge_rank"]

            if cap is None:
                cap = _HARDCODED_RANK_CAPS.get(unit_id, 7)

            session._rank_cap_cache[unit_id] = cap

        max_rank = session._rank_cap_cache[unit_id]
        return min(rank, max_rank) if max_rank is not None else rank

    # ------------------------------------------------------------------
    # Stage 4 — Hero identification
    # ------------------------------------------------------------------

    def _process_heroes(self,
                         frame,
                         session: MatchSession,
                         state: GameState) -> None:
        """
        Classify hero portraits and cache results.

        Runs on the first frame and every _HERO_SCAN_INTERVAL frames thereafter
        to handle VODs where the player portrait is not visible at frame 0.
        """
        should_scan = (
            session.frame_count == 1
            or session.frame_count % _HERO_SCAN_INTERVAL == 0
            or (session.player_hero_id is None or session.opponent_hero_id is None)
        )

        if should_scan:
            results = self._hero.classify_frame(frame)
            if results["player"] is not None:
                session.player_hero_id = results["player"].hero_id
            if results["opponent"] is not None:
                session.opponent_hero_id = results["opponent"].hero_id

        state.player_hero_id   = session.player_hero_id
        state.opponent_hero_id = session.opponent_hero_id

    # ------------------------------------------------------------------
    # Stage 3.5 — Animation detection
    # ------------------------------------------------------------------

    def _process_animations(self,
                             frame,
                             calibrator: GridCalibrator,
                             state: GameState) -> None:
        """
        Detect active buff/debuff animations on all occupied cells.

        Populates state.player_active_buffs and state.opponent_active_buffs
        ({unit_id: [animation_ids]}) and merges both into state.active_buffs
        for snapshot serialisation compatibility.
        """
        for side, board, buffs_dict in [
            ("player",   state.player_board,   state.player_active_buffs),
            ("opponent", state.opponent_board, state.opponent_active_buffs),
        ]:
            for row, col, cell in board.occupied():
                crop = calibrator.crop_cell(frame, side, row, col)
                anims = self._animations.detect(crop, cell.unit_id)
                if anims:
                    buffs_dict[cell.unit_id] = anims

        state.active_buffs = {
            **state.player_active_buffs,
            **state.opponent_active_buffs,
        }

    # ------------------------------------------------------------------
    # Stage 5 — HUD OCR
    # ------------------------------------------------------------------

    def _process_hud(self, frame, state: GameState) -> None:
        """Read wave, HP, and summon button state from the HUD."""
        readings = self._ocr.read(frame)
        state.wave_number  = readings.wave_number
        state.player_hp    = readings.player_hp
        state.opponent_hp  = readings.opponent_hp
        state.summon_ready = readings.summon_ready


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _db_lookup_fixed_branch(db_conn,
                             unit_id: str,
                             tier: int) -> Optional[str]:
    """
    Return 'Fixed' if this tier is Fixed for the given unit, else None.

    A Fixed tier has exactly one row in talent_trees with branch='Fixed'.
    A branching tier has rows with branch='L' and/or branch='R' instead.
    """
    row = db_conn.execute(
        "SELECT branch FROM talent_trees "
        "WHERE unit_id = ? AND tier = ? AND branch = 'Fixed'",
        (unit_id, tier)
    ).fetchone()
    return "Fixed" if row is not None else None
