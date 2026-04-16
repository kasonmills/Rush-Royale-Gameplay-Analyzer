"""
MatchRunner — end-to-end match analysis pipeline.

Ties together every Phase 1 and Phase 2 module into a single entry point:

  Capture  →  GridCalibrator  →  MCR  →  WinPredictor  →  match_history.db

Two modes:
  Video file — analyses a recorded VOD, sampling at a configurable rate.
  Live capture — grabs the desktop window (or scrcpy) in real time.

The runner handles:
  - Auto-initialising the three SQLite databases if they don't exist yet.
  - Loading all recognition modules from the asset directories.
  - Building a GridCalibrator from a saved calibration or sensible defaults.
  - Running the frame loop and calling the MCR on each sampled frame.
  - Throttled snapshot writes to match_history.db (configurable interval).
  - Unit performance aggregation written at match end.
  - Optional per-frame callback for UI / overlay consumption.

Usage — video file:
    runner = MatchRunner.for_video("recording.mp4")
    result = runner.run()
    print(f"Match ID: {result.match_id}")
    runner.set_outcome("win")   # call once you know the result

Usage — live desktop window:
    runner = MatchRunner.for_window("Rush Royale")
    result = runner.run(on_state=lambda s: print(s.win_probability))
    runner.set_outcome("loss")

Usage — custom config:
    cfg = MatchRunnerConfig(sample_every_sec=1.0, snapshot_interval_sec=5.0)
    runner = MatchRunner.for_video("vod.mp4", config=cfg)
    result = runner.run()
"""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from src.analysis.game_state import GameState
from src.analysis.match_context_resolver import MatchContextResolver, MatchSession
from src.analysis.win_predictor import WinPredictor
from src.capture.grid_calibrator import GridCalibrator
from src.capture.screen_capture import ScrcpyCapture, WindowCapture
from src.capture.video_capture import VideoCapture
from src.database.match_history_repo import MatchRepo, SnapshotRepo, UnitPerformanceRepo
from src.recognition.hero_classifier import HeroClassifier
from src.recognition.ocr_reader import OCRReader
from src.recognition.talent_classifier import TalentClassifier
from src.recognition.template_matcher import TemplateMatcher


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Snapshot confidence below which the frame is not written to the DB.
# The MCR still processes it — this only gates DB persistence.
_MIN_PERSIST_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MatchRunnerConfig:
    """
    All tuneable parameters for a MatchRunner.

    Any path left as None defaults to the standard project layout under
    the repository root (assets/reference, data/calibration.json, etc.).
    """

    # Asset directories
    reference_dir:    Optional[Path] = None  # assets/reference
    talent_icon_dir:  Optional[Path] = None  # assets/reference/talent_icons
    hero_portrait_dir: Optional[Path] = None  # assets/reference/hero_portraits

    # Calibration
    calibration_path: Optional[Path] = None  # data/calibration.json

    # Frame sampling
    sample_every_sec: float = 0.5   # video mode: one frame per this many seconds
    live_fps: float = 5.0           # live mode: capture rate (frames per second)

    # DB persistence
    persist: bool = True            # False = run without writing anything to disk
    snapshot_interval_sec: float = 2.0  # min match-seconds between snapshot writes

    # Frame normalisation (all frames are scaled to this width before processing)
    target_width: int = 1080

    def __post_init__(self):
        root = _PROJECT_ROOT
        if self.reference_dir is None:
            self.reference_dir = root / "assets" / "reference"
        if self.talent_icon_dir is None:
            self.talent_icon_dir = root / "assets" / "reference" / "talent_icons"
        if self.hero_portrait_dir is None:
            self.hero_portrait_dir = root / "assets" / "reference" / "hero_portraits"
        if self.calibration_path is None:
            self.calibration_path = root / "data" / "calibration.json"

        # Ensure they are Path objects even if the caller passed strings
        self.reference_dir    = Path(self.reference_dir)
        self.talent_icon_dir  = Path(self.talent_icon_dir)
        self.hero_portrait_dir = Path(self.hero_portrait_dir)
        self.calibration_path = Path(self.calibration_path)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Summary returned by MatchRunner.run()."""

    match_id: str
    source_type: str                    # 'video_file' | 'live_capture'
    source_path: Optional[str]          # video path, or None for live

    player_deck:     set[str] = field(default_factory=set)
    opponent_deck:   set[str] = field(default_factory=set)
    player_hero_id:  Optional[str] = None
    opponent_hero_id: Optional[str] = None

    total_frames_processed: int = 0
    total_snapshots_written: int = 0
    duration_sec: float = 0.0

    final_wave:            Optional[int]   = None
    final_win_probability: Optional[float] = None


# ---------------------------------------------------------------------------
# MatchRunner
# ---------------------------------------------------------------------------

class MatchRunner:
    """
    Orchestrates one match analysis session (video file or live capture).

    Create with the factory classmethods for_video(), for_window(), or
    for_scrcpy().  Then call run() once.  Optionally call set_outcome()
    afterwards to tag the match result in the DB.
    """

    def __init__(self,
                 source_type: str,
                 source,
                 source_path: Optional[str] = None,
                 config: Optional[MatchRunnerConfig] = None):
        self._source_type = source_type
        self._source      = source   # VideoCapture, WindowCapture, or ScrcpyCapture
        self._source_path = source_path
        self._config      = config or MatchRunnerConfig()
        self._last_match_id: Optional[str] = None
        self._stop_requested: bool = False

        # Recognition modules — populated by _setup_recognizers()
        self._matcher:   Optional[TemplateMatcher]  = None
        self._talent:    Optional[TalentClassifier] = None
        self._hero:      Optional[HeroClassifier]   = None
        self._ocr:       OCRReader = OCRReader()
        self._predictor: WinPredictor = WinPredictor()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_video(cls,
                  path: str | Path,
                  config: Optional[MatchRunnerConfig] = None) -> "MatchRunner":
        """
        Analyse a recorded video file.  Game region is auto-detected;
        frames are sampled at config.sample_every_sec.
        """
        cfg = config or MatchRunnerConfig()
        cap = VideoCapture(path, target_width=cfg.target_width)
        return cls("video_file", cap, source_path=str(path), config=cfg)

    @classmethod
    def for_window(cls,
                   window_title: str = "Rush Royale",
                   config: Optional[MatchRunnerConfig] = None) -> "MatchRunner":
        """
        Capture the Rush Royale desktop window in real time.
        """
        cfg = config or MatchRunnerConfig()
        cap = WindowCapture(window_title=window_title, target_width=cfg.target_width)
        return cls("live_capture", cap, source_path=None, config=cfg)

    @classmethod
    def for_scrcpy(cls,
                   width: int = 1080,
                   max_fps: int = 30,
                   serial: Optional[str] = None,
                   config: Optional[MatchRunnerConfig] = None) -> "MatchRunner":
        """
        Capture from a connected Android device via scrcpy.
        """
        cfg = config or MatchRunnerConfig()
        cap = ScrcpyCapture(width=width, max_fps=max_fps, serial=serial)
        return cls("live_capture", cap, source_path=None, config=cfg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self,
            on_state: Optional[Callable[[GameState], None]] = None) -> MatchResult:
        """
        Run the full analysis pipeline for one match.

        Args:
            on_state: Optional callback invoked after each processed frame
                      with the populated GameState.  Use this to drive a
                      UI overlay or log custom metrics.

        Returns:
            MatchResult with deck IDs, hero IDs, and summary stats.
        """
        cfg = self._config

        # Ensure databases exist before anything else
        _ensure_databases()

        # Load all recognition modules
        self._setup_recognizers()

        match_id  = str(uuid.uuid4())
        self._last_match_id = match_id
        result = MatchResult(
            match_id=match_id,
            source_type=self._source_type,
            source_path=self._source_path,
        )

        print(f"[MatchRunner] Starting match {match_id} "
              f"({self._source_type}: {self._source_path or 'live'})")

        with self._source as src:
            # Build calibrator from the first available frame
            calibrator = self._setup_calibrator(src)
            if calibrator is None:
                print("[MatchRunner] ERROR: could not read first frame — aborting.")
                return result

            # Build MCR and session
            mcr     = MatchContextResolver(self._matcher, self._talent,
                                           self._hero, self._ocr)
            session = mcr.start_match(match_id)

            # Open DB connections for the duration of the loop
            meta_conn, mh_conn = _open_connections(cfg.persist)

            try:
                if cfg.persist:
                    _write_initial_match_record(mh_conn, match_id,
                                                self._source_type,
                                                self._source_path)

                last_snapshot_time = -999.0   # ensure first frame is written
                match_start_wall   = time.monotonic()

                for frame, timestamp_sec in self._frame_iterator(src):
                    state = mcr.process_frame(
                        frame, calibrator, session, timestamp_sec,
                        db_conn=meta_conn,
                    )
                    self._predictor.predict(state, db_conn=meta_conn)

                    result.total_frames_processed += 1

                    if on_state is not None:
                        on_state(state)

                    # Throttled snapshot persist
                    if (cfg.persist
                            and mh_conn is not None
                            and state.pipeline_confidence >= _MIN_PERSIST_CONFIDENCE
                            and timestamp_sec - last_snapshot_time >= cfg.snapshot_interval_sec):
                        SnapshotRepo.insert(mh_conn, state.to_snapshot_dict())
                        result.total_snapshots_written += 1
                        last_snapshot_time = timestamp_sec

                # ---- Match end ----
                result.duration_sec          = time.monotonic() - match_start_wall
                result.player_deck           = session.player_deck
                result.opponent_deck         = session.opponent_deck
                result.player_hero_id        = session.player_hero_id
                result.opponent_hero_id      = session.opponent_hero_id
                last = session.last_state
                result.final_wave            = last.wave_number if last else None
                result.final_win_probability = last.win_probability if last else None

                if cfg.persist and mh_conn is not None:
                    _write_unit_performance(mh_conn, match_id, session)
                    _finalise_match_record(mh_conn, match_id, result)
                    mh_conn.commit()
                    print(f"[MatchRunner] Wrote {result.total_snapshots_written} snapshots "
                          f"for match {match_id}.")

            finally:
                if meta_conn:
                    meta_conn.close()
                if mh_conn:
                    mh_conn.commit()
                    mh_conn.close()

        print(f"[MatchRunner] Done. {result.total_frames_processed} frames processed "
              f"in {result.duration_sec:.1f}s.")
        return result

    def stop(self):
        """
        Request the running match to stop gracefully.

        The current frame finishes processing; the loop then exits after
        the next frame is yielded.  Safe to call from any thread.
        """
        self._stop_requested = True

    def set_outcome(self, outcome: str):
        """
        Record the match outcome after run() has returned.

        Args:
            outcome: 'win' or 'loss'  (from the player's perspective)
        """
        if self._last_match_id is None:
            raise RuntimeError("No match has been run yet — call run() first.")
        if outcome not in ("win", "loss"):
            raise ValueError(f"outcome must be 'win' or 'loss', got {outcome!r}")

        meta_conn, mh_conn = _open_connections(self._config.persist)
        try:
            if mh_conn is not None:
                MatchRepo.set_outcome(mh_conn, self._last_match_id, outcome)
                mh_conn.commit()
                print(f"[MatchRunner] Outcome '{outcome}' recorded for "
                      f"match {self._last_match_id}.")
        finally:
            if meta_conn:
                meta_conn.close()
            if mh_conn:
                mh_conn.close()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_recognizers(self):
        """Load all recognition modules from the configured asset directories."""
        cfg = self._config

        self._matcher = TemplateMatcher()
        if cfg.reference_dir.is_dir():
            self._matcher.load_library(cfg.reference_dir)
        else:
            print(f"[MatchRunner] WARNING: reference directory not found at "
                  f"{cfg.reference_dir} — template matching will not work.")
            self._matcher._loaded = True   # prevent load() assertion errors

        self._talent = TalentClassifier()
        if cfg.talent_icon_dir.is_dir():
            self._talent.load(cfg.talent_icon_dir)
        else:
            print(f"[MatchRunner] WARNING: talent icon directory not found at "
                  f"{cfg.talent_icon_dir} — talent classification disabled.")
            self._talent._loaded = True

        self._hero = HeroClassifier()
        if cfg.hero_portrait_dir.is_dir():
            self._hero.load(cfg.hero_portrait_dir)
        else:
            print(f"[MatchRunner] WARNING: hero portrait directory not found at "
                  f"{cfg.hero_portrait_dir} — hero classification disabled.")
            self._hero._loaded = True

    def _setup_calibrator(self, src) -> Optional[GridCalibrator]:
        """
        Build a GridCalibrator for this stream.

        Tries loading a saved calibration first.  Falls back to defaults
        computed from the first captured frame's dimensions.
        """
        cfg = self._config

        # Try saved calibration
        if cfg.calibration_path.exists():
            try:
                cal = GridCalibrator.load(cfg.calibration_path)
                print(f"[MatchRunner] Loaded calibration from {cfg.calibration_path}")
                return cal
            except Exception as e:
                print(f"[MatchRunner] WARNING: could not load calibration "
                      f"({e}) — using defaults.")

        # Probe first frame to get dimensions
        first_frame = self._get_first_frame(src)
        if first_frame is None:
            return None

        fh, fw = first_frame.shape[:2]
        cal = GridCalibrator.from_defaults(fw, fh)
        print(f"[MatchRunner] Using default calibration for {fw}×{fh} frame.")
        return cal

    def _get_first_frame(self, src):
        """Grab the first frame from the source without consuming the iterator."""
        if isinstance(src, VideoCapture):
            # Auto-detect game region then read frame 0
            src.detect_game_region()
            return src.frame_at(0.0)
        if isinstance(src, WindowCapture):
            return src.grab()
        if isinstance(src, ScrcpyCapture):
            # ScrcpyCapture needs a moment to initialise; latest_frame() may be
            # None right after start — wait briefly then try
            for _ in range(20):
                frame = src.latest_frame()
                if frame is not None:
                    return frame
                time.sleep(0.1)
            return None
        return None

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def _frame_iterator(
            self, src) -> Iterator[tuple]:
        """
        Unified frame iterator that yields (frame, timestamp_sec) regardless
        of the underlying source type.

        - VideoCapture: delegates to src.frames(sample_every=…) which already
          returns (frame, timestamp_sec) from the video clock.
        - WindowCapture / ScrcpyCapture: yields at live_fps, tracking elapsed
          wall-clock time as the timestamp.
        """
        cfg = self._config

        if isinstance(src, VideoCapture):
            for frame, ts in src.frames(sample_every=cfg.sample_every_sec):
                if self._stop_requested:
                    return
                yield frame, ts
            return

        # Live source — derive timestamps from wall clock.
        # WindowCapture.frames() and ScrcpyCapture.frames() both pace themselves
        # internally, so no extra sleep is needed here.
        start = time.monotonic()
        for frame in src.frames(target_fps=cfg.live_fps):
            if self._stop_requested:
                return
            yield frame, time.monotonic() - start


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_databases():
    """Create all three databases if they don't exist yet."""
    from src.database.init_db import init_all
    from src.database.connection import _DB_PATHS
    if not all(p.exists() for p in _DB_PATHS.values()):
        print("[MatchRunner] Initialising databases...")
        init_all()


def _open_connections(persist: bool) -> tuple[Optional[sqlite3.Connection],
                                               Optional[sqlite3.Connection]]:
    """
    Open raw sqlite3 connections to unit_meta.db and match_history.db.

    Returns (meta_conn, mh_conn).  Both are None when persist=False.
    The caller owns the connections and must close them.
    """
    if not persist:
        return None, None

    from src.database.connection import _DB_PATHS

    meta_conn = sqlite3.connect(_DB_PATHS["unit_meta"])
    meta_conn.row_factory = sqlite3.Row
    meta_conn.execute("PRAGMA foreign_keys = ON")

    mh_conn = sqlite3.connect(_DB_PATHS["match_history"])
    mh_conn.row_factory = sqlite3.Row
    mh_conn.execute("PRAGMA foreign_keys = ON")

    return meta_conn, mh_conn


def _write_initial_match_record(conn: sqlite3.Connection,
                                 match_id: str,
                                 source_type: str,
                                 source_path: Optional[str]):
    """Insert the matches row at match start with fields known immediately."""
    MatchRepo.insert(conn, {
        "match_id":    match_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_path": source_path,
    })
    conn.commit()


def _finalise_match_record(conn: sqlite3.Connection,
                            match_id: str,
                            result: MatchResult):
    """Update the matches row with fields known only at match end."""
    conn.execute(
        """
        UPDATE matches
        SET player_hero_id   = ?,
            opponent_hero_id = ?,
            player_deck      = ?,
            opponent_deck    = ?,
            total_waves      = ?,
            match_duration_sec = ?
        WHERE match_id = ?
        """,
        (
            result.player_hero_id,
            result.opponent_hero_id,
            json.dumps(sorted(result.player_deck)),
            json.dumps(sorted(result.opponent_deck)),
            result.final_wave,
            round(result.duration_sec, 2),
            match_id,
        )
    )


def _write_unit_performance(conn: sqlite3.Connection,
                             match_id: str,
                             session: MatchSession):
    """
    Aggregate and write per-unit performance rows from the session.

    For each (player, unit_id) pair encountered during the match:
      - max_rank_seen:   highest merge_rank observed across all frames
      - talent_tier_seen: highest talent tier in the accumulated cache
      - talent_branch:   branch for that tier (if resolved)
      - branch_confidence: fraction of cached tiers that are resolved
    """
    rows: list[dict] = []

    # Gather max ranks from the last state
    last = session.last_state
    rank_map: dict[tuple[str, str], int] = {}
    if last is not None:
        for player_label, board in (("player", last.player_board),
                                    ("opponent", last.opponent_board)):
            for _, _, cell in board.occupied():
                key = (player_label, cell.unit_id)
                rank_map[key] = max(rank_map.get(key, 0), cell.merge_rank)

    for (player_label, unit_id), tier_cache in session.talent_cache.items():
        if not tier_cache:
            continue

        top_tier   = max(tier_cache.keys())
        top_branch = tier_cache.get(top_tier)

        resolved = sum(1 for b in tier_cache.values() if b is not None)
        b_conf   = resolved / len(tier_cache) if tier_cache else 1.0

        rows.append({
            "match_id":         match_id,
            "unit_id":          unit_id,
            "player":           player_label,
            "max_rank_seen":    rank_map.get((player_label, unit_id)),
            "talent_tier_seen": top_tier,
            "talent_branch":    top_branch,
            "branch_confidence": round(b_conf, 3),
        })

    # Also add units with rank data but no talent cache entries
    for (player_label, unit_id), max_rank in rank_map.items():
        if (player_label, unit_id) not in session.talent_cache:
            rows.append({
                "match_id":      match_id,
                "unit_id":       unit_id,
                "player":        player_label,
                "max_rank_seen": max_rank,
            })

    if rows:
        UnitPerformanceRepo.insert_many(conn, rows)
