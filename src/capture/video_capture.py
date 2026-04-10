"""
Video file analysis via OpenCV VideoCapture.

Recorded Rush Royale footage is rarely a clean game-only crop. Streamers
typically have overlays, face cams, chat panels, or letterboxing. This module
handles that by:

  1. Letting the caller specify the game region explicitly (most reliable).
  2. Providing auto-detection that searches for the game within the frame using
     the expected portrait aspect ratio and UI colour cues.
  3. Falling back to full-frame if neither is available (caller then passes
     frames to GridCalibrator which further crops to the board).

Usage:
    with VideoCapture("vod.mp4") as cap:
        # Let the auto-detector find the game region in the first few frames
        cap.detect_game_region()
        print("Game region:", cap.game_region)

        for frame, timestamp_sec in cap.frames(sample_every=0.5):
            # frame is already cropped to the detected game region
            process(frame, timestamp_sec)

    # Or specify the region manually (pixels in the raw video frame):
    with VideoCapture("vod.mp4", game_region=(x, y, w, h)) as cap:
        for frame, ts in cap.frames():
            ...
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


# Expected game aspect ratio (width / height) for Rush Royale portrait layout.
# Phone native is ~9:19.5 ≈ 0.46; desktop may differ but is still portrait.
_GAME_ASPECT_MIN = 0.40
_GAME_ASPECT_MAX = 0.65


class VideoCapture:
    """
    Thin wrapper around cv2.VideoCapture with game-region awareness.

    Args:
        source:       Path to a video file, or an integer device index.
        game_region:  (x, y, w, h) in raw frame pixels. If provided, all
                      yielded frames are pre-cropped to this region.
        target_width: After cropping, resize frames to this width (preserves
                      aspect ratio). Normalises resolution before recognition.
    """

    def __init__(self,
                 source: str | Path | int,
                 game_region: Optional[tuple[int, int, int, int]] = None,
                 target_width: Optional[int] = None):
        self._source = source
        self.game_region = game_region   # (x, y, w, h) — mutable; set by detect_game_region()
        self._target_width = target_width
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoCapture":
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def open(self):
        src = str(self._source) if isinstance(self._source, Path) else self._source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self._source!r}")

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        self._require_open()
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_sec(self) -> float:
        fps = self.fps
        return self.frame_count / fps if fps > 0 else 0.0

    @property
    def raw_frame_size(self) -> tuple[int, int]:
        """(width, height) of the full unprocessed frame."""
        self._require_open()
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    # ------------------------------------------------------------------
    # Game region detection
    # ------------------------------------------------------------------

    def detect_game_region(self, sample_count: int = 5) -> Optional[tuple[int, int, int, int]]:
        """
        Samples `sample_count` frames spread through the first 20% of the video
        and attempts to locate the Rush Royale game within each frame using two
        heuristics (tried in order):

          1. Largest portrait-aspect-ratio rectangle found via contour analysis
             on the game's distinctive dark border / background.
          2. Known UI colour band — Rush Royale has a characteristic dark
             header/footer band above and below the play area; detected by
             scanning horizontal luminance profiles.

        Sets ``self.game_region`` to the best (x, y, w, h) found and returns it.
        Returns None if detection fails (caller should specify region manually).
        """
        self._require_open()
        # Sample across first 20% of video (avoids pre-game lobby at the end)
        sample_positions = [
            int(self.frame_count * 0.05 * i) for i in range(1, sample_count + 1)
        ]

        candidates: list[tuple[int, int, int, int]] = []
        for pos in sample_positions:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = self._cap.read()
            if not ok:
                continue
            region = _detect_game_region_in_frame(frame)
            if region is not None:
                candidates.append(region)

        # Rewind to start
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not candidates:
            print("[VideoCapture] Game region auto-detection failed. "
                  "Set video_capture.game_region = (x, y, w, h) manually.")
            return None

        # Pick the most common candidate (or average if all unique)
        region = _consensus_region(candidates)
        print(f"[VideoCapture] Detected game region: x={region[0]} y={region[1]} "
              f"w={region[2]} h={region[3]}")
        self.game_region = region
        return region

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def frames(self,
               sample_every: float = 0.5,
               start_sec: float = 0.0,
               end_sec: Optional[float] = None) -> Iterator[tuple[np.ndarray, float]]:
        """
        Yields (frame, timestamp_sec) sampled every `sample_every` seconds.

        Each frame is:
          1. Cropped to self.game_region if set (removes non-game content).
          2. Resized to target_width if set.

        Args:
            sample_every: Seconds between yielded frames.
            start_sec:    Skip to this timestamp before starting.
            end_sec:      Stop at this timestamp (None = until end of video).
        """
        self._require_open()
        fps = self.fps
        step_frames = max(1, int(fps * sample_every))

        if start_sec > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

        while True:
            pos_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = pos_frame / fps

            if end_sec is not None and timestamp > end_sec:
                break

            ok, frame = self._cap.read()
            if not ok:
                break

            yield self._process_frame(frame), timestamp

            next_frame = pos_frame + step_frames
            if next_frame >= self.frame_count:
                break
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    def frame_at(self, timestamp_sec: float) -> Optional[np.ndarray]:
        """Returns a single processed frame at the given timestamp, or None."""
        self._require_open()
        target = int(timestamp_sec * self.fps)
        if target >= self.frame_count:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self._cap.read()
        return self._process_frame(frame) if ok else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop to game_region then resize."""
        if self.game_region is not None:
            x, y, w, h = self.game_region
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            frame = frame[y1:y2, x1:x2]
        if self._target_width is not None:
            h, w = frame.shape[:2]
            if w != self._target_width:
                scale = self._target_width / w
                frame = cv2.resize(frame, (self._target_width, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        return frame

    def _require_open(self):
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                "VideoCapture is not open. Use as a context manager or call open() first."
            )


# ---------------------------------------------------------------------------
# Game region detection helpers
# ---------------------------------------------------------------------------

def _detect_game_region_in_frame(
        frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Attempt to locate the Rush Royale game area within a single frame.

    Strategy:
      - Convert to grayscale and threshold to isolate the dark game background.
      - Find the largest contour with a portrait aspect ratio.
      - Validate that the candidate occupies at least 15% of the total frame area.
    """
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rush Royale game background is dark; non-game areas tend to be lighter
    # (stream backgrounds, chat panels). Threshold to isolate dark regions.
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Morphological close to fill small gaps inside the game area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = fw * fh * 0.15  # at least 15% of frame
    best: Optional[tuple[int, int, int, int]] = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        aspect = w / h if h > 0 else 0
        if _GAME_ASPECT_MIN <= aspect <= _GAME_ASPECT_MAX and area > best_area:
            best = (x, y, w, h)
            best_area = area

    return best


def _consensus_region(
        regions: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """
    Given several candidate regions, return a consensus by averaging them.
    Rounds to nearest integer.
    """
    xs = [r[0] for r in regions]
    ys = [r[1] for r in regions]
    ws = [r[2] for r in regions]
    hs = [r[3] for r in regions]
    return (
        int(round(sum(xs) / len(xs))),
        int(round(sum(ys) / len(ys))),
        int(round(sum(ws) / len(ws))),
        int(round(sum(hs) / len(hs))),
    )