"""
Live screen capture — desktop window or Android mirror (scrcpy).

Rush Royale runs natively on desktop, so the primary path is capturing the
game window directly. scrcpy remains an option for phone sessions.

Two backends:
  WindowCapture  — grabs a named desktop window or a manually specified region
                   using mss. Works for the Rush Royale desktop app or an emulator.
  ScrcpyCapture  — streams frames from an Android device via a scrcpy subprocess.

In both cases, callers receive BGR numpy arrays. The captured area may contain
UI chrome outside the game; use GridCalibrator to locate the board within it.

Usage — desktop window:
    with WindowCapture(window_title="Rush Royale") as cap:
        for frame in cap.frames(target_fps=10):
            process(frame)

    # Or specify a region manually:
    with WindowCapture(region={"top": 50, "left": 100, "width": 600, "height": 900}) as cap:
        ...

Usage — scrcpy (phone mirror):
    with ScrcpyCapture(width=1080, height=2340) as cap:
        for frame in cap.frames(target_fps=10):
            process(frame)
"""

import subprocess
import threading
import time
from typing import Iterator, Optional

import cv2
import numpy as np

try:
    import mss
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False


# ---------------------------------------------------------------------------
# WindowCapture — desktop (primary path)
# ---------------------------------------------------------------------------

class WindowCapture:
    """
    Captures a desktop window or screen region via mss.

    If `window_title` is given, attempts to locate the window's bounding box
    automatically on Windows using pygetwindow (optional dep). Falls back to
    full primary monitor capture if the window can't be found.

    If `region` is given explicitly, it takes priority over window auto-detect.
    region format: {"top": y, "left": x, "width": w, "height": h}

    Args:
        window_title:  Partial title match for the game window.
        region:        Explicit pixel region dict (overrides window_title lookup).
        target_width:  Resize frames to this width after capture (preserves aspect).
    """

    def __init__(self,
                 window_title: Optional[str] = None,
                 region: Optional[dict] = None,
                 target_width: Optional[int] = None):
        if not _MSS_AVAILABLE:
            raise ImportError(
                "mss is required. Install with: pip install mss"
            )
        self._window_title = window_title
        self._region = region
        self._target_width = target_width
        self._sct = None

    def __enter__(self) -> "WindowCapture":
        self._sct = mss.mss()
        if self._region is None and self._window_title:
            self._region = _find_window_region(self._window_title)
        return self

    def __exit__(self, *_):
        if self._sct:
            self._sct.close()
            self._sct = None

    def grab(self) -> np.ndarray:
        """Capture and return a single BGR frame."""
        if self._sct is None:
            raise RuntimeError("WindowCapture not open — use as context manager.")
        monitor = self._region or self._sct.monitors[1]
        shot = self._sct.grab(monitor)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return _maybe_resize(frame, self._target_width)

    def frames(self, target_fps: float = 10.0) -> Iterator[np.ndarray]:
        """Yields frames at up to `target_fps` per second."""
        interval = 1.0 / target_fps
        while True:
            start = time.monotonic()
            yield self.grab()
            elapsed = time.monotonic() - start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    @property
    def region(self) -> Optional[dict]:
        """The resolved capture region (set after open())."""
        return self._region


# ---------------------------------------------------------------------------
# ScrcpyCapture — phone mirror
# ---------------------------------------------------------------------------

class ScrcpyCapture:
    """
    Launches scrcpy with --no-display and reads raw frames from its stdout pipe.

    Prerequisites:
        - scrcpy on PATH (https://github.com/Genymobile/scrcpy)
        - Device connected over ADB (USB or TCP/IP)

    Args:
        width:       Target output width scrcpy should scale the device to.
        max_fps:     Frame rate cap forwarded to scrcpy.
        serial:      ADB device serial for multi-device setups.
        scrcpy_path: Override if scrcpy is not on PATH.
    """

    def __init__(self,
                 width: int = 1080,
                 max_fps: int = 30,
                 serial: Optional[str] = None,
                 scrcpy_path: str = "scrcpy"):
        self.width = width
        self.max_fps = max_fps
        self.serial = serial
        self._scrcpy = scrcpy_path
        self._proc: Optional[subprocess.Popen] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ScrcpyCapture":
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        cmd = [
            self._scrcpy,
            "--no-display",
            f"--max-size={self.width}",
            f"--max-fps={self.max_fps}",
            "--video-codec=raw",
        ]
        if self.serial:
            cmd += ["--serial", self.serial]

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_frames, daemon=True
        )
        self._reader_thread.start()
        time.sleep(1.0)  # let scrcpy initialise

    def stop(self):
        self._running = False
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            self._proc = None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def frames(self, target_fps: float = 10.0) -> Iterator[np.ndarray]:
        interval = 1.0 / target_fps
        while self._running:
            frame = self.latest_frame()
            if frame is not None:
                yield frame
            time.sleep(interval)

    def _read_frames(self):
        """Background thread: reads raw RGB24 frames from scrcpy stdout."""
        MAGIC = b"\x00\x00\x00\x01"

        while self._running and self._proc and self._proc.poll() is None:
            try:
                header = self._proc.stdout.read(12)
                if len(header) < 12:
                    break
                # Re-sync if magic not found
                if header[:4] != MAGIC:
                    buf = header
                    while self._running:
                        buf = buf[1:] + self._proc.stdout.read(1)
                        if buf[:4] == MAGIC:
                            self._proc.stdout.read(8)
                            break
                    continue

                w = int.from_bytes(header[4:8], "big")
                h = int.from_bytes(header[8:12], "big")
                raw = self._proc.stdout.read(w * h * 3)
                if len(raw) < w * h * 3:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._frame = frame

            except (OSError, ValueError):
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_window_region(title: str) -> Optional[dict]:
    """
    Tries to locate a desktop window by partial title match and returns its
    screen region dict. Returns None if pygetwindow is unavailable or the
    window is not found.
    """
    try:
        import pygetwindow as gw
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            print(f"[WindowCapture] No window found matching '{title}'. "
                  "Capturing full primary monitor.")
            return None
        win = matches[0]
        return {
            "top": max(0, win.top),
            "left": max(0, win.left),
            "width": win.width,
            "height": win.height,
        }
    except ImportError:
        print("[WindowCapture] pygetwindow not installed — capturing full monitor. "
              "Install with: pip install pygetwindow")
        return None


def _maybe_resize(frame: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)