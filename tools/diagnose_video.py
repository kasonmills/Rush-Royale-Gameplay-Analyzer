"""
Diagnostic tool: visualise all calibrated pipeline regions on sample frames.

Draws every region the pipeline uses — board cell grids, deck strips, summon
strip, HUD (wave / HP) regions, and hero portrait regions — on frames sampled
from a video file, then saves the annotated images to data/screenshots/hud_debug/.

Use this after changing calibration constants to visually confirm that the
boxes land in the right places before running a full match analysis.

Usage:
    .venv\\Scripts\\python.exe tools\\diagnose_video.py
    .venv\\Scripts\\python.exe tools\\diagnose_video.py path\\to\\video.mp4
    .venv\\Scripts\\python.exe tools\\diagnose_video.py --pct 0.1 0.3 0.5 0.7
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.capture.grid_calibrator import GridCalibrator
from src.capture.video_capture import VideoCapture
from src.recognition.hero_classifier import HUDRegions
from src.recognition.ocr_reader import HUDLayout

_DEFAULT_VIDEO = _ROOT / "data" / "screenshots" / "hud_frames" / "gameplay.mp4"
_OUT_DIR       = _ROOT / "data" / "screenshots" / "hud_debug"
_DEFAULT_PCTS  = [0.10, 0.25, 0.50, 0.75]

# BGR colour palette
_C_PLAYER_BOARD = (0,   220,   0)   # green
_C_OPP_BOARD    = (0,   140, 255)   # orange
_C_PLAYER_DECK  = (220, 220,   0)   # cyan
_C_OPP_DECK     = (200, 200,   0)   # darker cyan
_C_SUMMON       = (220,   0, 220)   # magenta
_C_WAVE         = (0,   220, 220)   # yellow
_C_HP           = (0,   180, 255)   # amber
_C_HERO         = (255, 100, 200)   # pink


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
          color: tuple, label: str = "", thickness: int = 2) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1 + 3, y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _frac_rect(img: np.ndarray,
               left_f: float, top_f: float, right_f: float, bot_f: float,
               color: tuple, label: str = "", thickness: int = 2) -> None:
    h, w = img.shape[:2]
    _rect(img, int(w * left_f), int(h * top_f),
          int(w * right_f), int(h * bot_f), color, label, thickness)


def _gridrect(img: np.ndarray, gridrect, color: tuple, label: str = "") -> None:
    """Draw a GridRect box with a label."""
    x, y, w, h = gridrect.x, gridrect.y, gridrect.w, gridrect.h
    _rect(img, x, y, x + w, y + h, color, label)


def _deck_slots(img: np.ndarray, gridrect, n_slots: int, color: tuple) -> None:
    """Draw vertical dividers for deck slot boundaries inside a GridRect."""
    x, y, w, h = gridrect.x, gridrect.y, gridrect.w, gridrect.h
    slot_w = w // n_slots
    for i in range(1, n_slots):
        sx = x + i * slot_w
        cv2.line(img, (sx, y), (sx, y + h), color, 1)


# ---------------------------------------------------------------------------
# Main annotation
# ---------------------------------------------------------------------------

def annotate(frame: np.ndarray, calibrator: GridCalibrator,
             hud: HUDLayout, heroes: HUDRegions) -> np.ndarray:
    """Return a copy of frame with all pipeline regions annotated."""
    out = frame.copy()
    data = calibrator._data

    # ---- Board cell grids ----
    for side in ("player", "opponent"):
        color = _C_PLAYER_BOARD if side == "player" else _C_OPP_BOARD
        label = "PLAYER BOARD" if side == "player" else "OPP BOARD"
        board = data.player_board if side == "player" else data.opponent_board
        # Outer box
        _gridrect(out, board, color, label)
        # Individual cells
        for row in range(data.rows):
            for col in range(data.cols):
                x, y, cw, ch = calibrator._cell_rect(side, row, col)
                cv2.rectangle(out, (x, y), (x + cw, y + ch), color, 1)
                cv2.putText(out, f"{row},{col}", (x + 2, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1, cv2.LINE_AA)

    # ---- Deck strips ----
    _gridrect(out, data.player_deck, _C_PLAYER_DECK, "PLAYER DECK")
    _deck_slots(out, data.player_deck, 5, _C_PLAYER_DECK)

    _gridrect(out, data.opponent_deck, _C_OPP_DECK, "OPP DECK")
    _deck_slots(out, data.opponent_deck, 5, _C_OPP_DECK)

    # ---- Summon strip ----
    _gridrect(out, data.player_summon, _C_SUMMON, "SUMMON")

    # ---- HUD regions ----
    _frac_rect(out, *hud.opponent_hp, _C_HP,   "OPP HP")
    _frac_rect(out, *hud.wave,        _C_WAVE,  "WAVE")
    _frac_rect(out, *hud.player_hp,   _C_HP,   "PLAYER HP")

    # ---- Hero portrait regions ----
    _frac_rect(out, *heroes.opponent_portrait, _C_HERO, "OPP HERO")
    _frac_rect(out, *heroes.player_portrait,   _C_HERO, "PLAYER HERO")

    return out


# ---------------------------------------------------------------------------
# Legend overlay
# ---------------------------------------------------------------------------

def _legend(img: np.ndarray) -> None:
    entries = [
        (_C_PLAYER_BOARD, "Player board cells"),
        (_C_OPP_BOARD,    "Opponent board cells"),
        (_C_PLAYER_DECK,  "Player deck slots"),
        (_C_OPP_DECK,     "Opponent deck slots"),
        (_C_SUMMON,       "Summon strip"),
        (_C_WAVE,         "Wave number (OCR)"),
        (_C_HP,           "HP regions (OCR)"),
        (_C_HERO,         "Hero portrait regions"),
    ]
    x0, y0 = 4, 16
    for i, (color, text) in enumerate(entries):
        y = y0 + i * 16
        cv2.rectangle(img, (x0, y - 10), (x0 + 10, y), color, -1)
        cv2.putText(img, text, (x0 + 14, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualise calibrated pipeline regions on sample frames."
    )
    ap.add_argument("video", nargs="?", default=str(_DEFAULT_VIDEO),
                    help="Path to video file (default: gameplay.mp4)")
    ap.add_argument("--pct", nargs="+", type=float, default=_DEFAULT_PCTS,
                    metavar="F",
                    help="Frame positions to sample as fractions of video "
                         "duration (default: 0.10 0.25 0.50 0.75)")
    ap.add_argument("--out-dir", default=str(_OUT_DIR),
                    help="Output directory (default: data/screenshots/hud_debug)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hud    = HUDLayout()
    heroes = HUDRegions()

    print(f"Video : {video_path}")
    print(f"Output: {out_dir}")
    print(f"Frames: {[f'{p:.0%}' for p in args.pct]}")
    print()

    with VideoCapture(video_path) as cap:
        cap.detect_game_region()

        first = cap.frame_at(0.0)
        if first is None:
            print("ERROR: could not read first frame.")
            sys.exit(1)

        fh, fw = first.shape[:2]
        duration = cap.duration_sec

        # Load saved calibration if present, else use defaults
        cal_path = _ROOT / "data" / "calibration.json"
        if cal_path.exists():
            calibrator = GridCalibrator.load(cal_path)
            print(f"Calibration: loaded from {cal_path}")
        else:
            calibrator = GridCalibrator.from_defaults(fw, fh)
            print(f"Calibration: defaults for {fw}×{fh}")

        print(f"Frame size: {fw}×{fh}  |  Duration: {duration:.1f}s")
        print()

        saved: list[str] = []
        for pct in args.pct:
            ts = pct * duration
            frame = cap.frame_at(ts)
            if frame is None:
                print(f"  SKIP  {pct:.0%}  (t={ts:.1f}s) — no frame returned")
                continue

            annotated = annotate(frame, calibrator, hud, heroes)
            _legend(annotated)

            name = f"diagnose_{int(pct * 100):03d}pct.png"
            path = out_dir / name
            cv2.imwrite(str(path), annotated)
            saved.append(str(path))
            print(f"  SAVED {pct:.0%}  (t={ts:.1f}s)  ->  {path.name}")

    print()
    if saved:
        print(f"Wrote {len(saved)} annotated frame(s) to {out_dir}")
        print("Open the images to verify that each coloured box lands on the")
        print("correct game element before running a full match analysis.")
    else:
        print("No frames were saved — check the video path and frame percentages.")


if __name__ == "__main__":
    main()