"""
GridCalibrator — defines the 3×5 board grid boundaries in pixel coordinates.

Rush Royale PvP layout (portrait phone screen):
  - Player board: bottom half of the frame, 3 cols × 5 rows
  - Opponent board: top half of the frame, 3 cols × 5 rows
  - Both boards share the same column layout; row order is mirrored
    (player row 0 = back/bottom, opponent row 0 = back/top)

Usage:
    cal = GridCalibrator.from_defaults(frame_width=1080, frame_height=2340)
    cell_img = cal.crop_cell(frame, player="player", row=2, col=1)

    # Or load a previously saved calibration:
    cal = GridCalibrator.load("data/calibration.json")

    # Interactive calibration (requires OpenCV window support):
    cal = GridCalibrator.interactive(frame)
    cal.save("data/calibration.json")
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default layout constants (tuned for a 1080×2340 scrcpy portrait stream)
# These are approximate starting points; interactive calibration refines them.
# ---------------------------------------------------------------------------

# Fraction of frame height where each board starts/ends
_PLAYER_BOARD_TOP_FRAC    = 0.55   # player board top edge
_PLAYER_BOARD_BOTTOM_FRAC = 0.92   # player board bottom edge
_OPP_BOARD_TOP_FRAC       = 0.08   # opponent board top edge
_OPP_BOARD_BOTTOM_FRAC    = 0.45   # opponent board bottom edge

# Deck icon strip — the 5 unit icons always visible below each board.
# Player deck icons sit just below the player board; opponent deck icons
# sit just above the opponent board.
_PLAYER_DECK_TOP_FRAC     = 0.93   # just below player board
_PLAYER_DECK_BOTTOM_FRAC  = 0.99
_OPP_DECK_TOP_FRAC        = 0.01   # just above opponent board
_OPP_DECK_BOTTOM_FRAC     = 0.07

# Fraction of frame width for the board left/right edges
_BOARD_LEFT_FRAC  = 0.04
_BOARD_RIGHT_FRAC = 0.96

DECK_SIZE = 5  # Rush Royale decks are always 5 units


@dataclass
class GridRect:
    """Pixel rectangle for one board: (x, y, width, height)."""
    x: int
    y: int
    w: int
    h: int

    def cell_rect(self, row: int, col: int, rows: int = 5, cols: int = 3
                  ) -> tuple[int, int, int, int]:
        """
        Returns (x, y, w, h) for a single cell within this board rect.
        row=0 is the TOP row of this rect (back row for player = lowest on screen,
        but back row for opponent = highest on screen — callers handle the mapping).
        """
        cell_w = self.w // cols
        cell_h = self.h // rows
        cx = self.x + col * cell_w
        cy = self.y + row * cell_h
        return cx, cy, cell_w, cell_h


@dataclass
class CalibrationData:
    frame_width: int
    frame_height: int
    player_board: GridRect
    opponent_board: GridRect
    player_deck: GridRect       # 5-icon strip below the player board
    opponent_deck: GridRect     # 5-icon strip above the opponent board
    rows: int = 5
    cols: int = 3


class GridCalibrator:
    """
    Holds calibrated board geometry and provides cell-crop helpers.
    """

    def __init__(self, data: CalibrationData):
        self._data = data

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(cls, frame_width: int, frame_height: int) -> "GridCalibrator":
        """
        Build a calibration from the hard-coded fractions.
        Good enough to get started; refine with interactive() if needed.
        """
        def frac_rect(top_f, bot_f, left_f=_BOARD_LEFT_FRAC,
                      right_f=_BOARD_RIGHT_FRAC) -> GridRect:
            x = int(frame_width * left_f)
            y = int(frame_height * top_f)
            w = int(frame_width * (right_f - left_f))
            h = int(frame_height * (bot_f - top_f))
            return GridRect(x, y, w, h)

        data = CalibrationData(
            frame_width=frame_width,
            frame_height=frame_height,
            player_board=frac_rect(_PLAYER_BOARD_TOP_FRAC, _PLAYER_BOARD_BOTTOM_FRAC),
            opponent_board=frac_rect(_OPP_BOARD_TOP_FRAC, _OPP_BOARD_BOTTOM_FRAC),
            player_deck=frac_rect(_PLAYER_DECK_TOP_FRAC, _PLAYER_DECK_BOTTOM_FRAC),
            opponent_deck=frac_rect(_OPP_DECK_TOP_FRAC, _OPP_DECK_BOTTOM_FRAC),
        )
        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> "GridCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        fw, fh = raw["frame_width"], raw["frame_height"]
        # Gracefully fall back to defaults for deck rects added after initial save
        def _rect(key):
            if key in raw:
                return GridRect(**raw[key])
            tmp = cls.from_defaults(fw, fh)
            return getattr(tmp._data, key)

        data = CalibrationData(
            frame_width=fw,
            frame_height=fh,
            player_board=_rect("player_board"),
            opponent_board=_rect("opponent_board"),
            player_deck=_rect("player_deck"),
            opponent_deck=_rect("opponent_deck"),
            rows=raw.get("rows", 5),
            cols=raw.get("cols", 3),
        )
        return cls(data)

    @classmethod
    def interactive(cls, frame: np.ndarray,
                    save_path: Optional[str | Path] = None) -> "GridCalibrator":
        """
        Opens an OpenCV window. User clicks four corners for each board
        (top-left then bottom-right of the board area) to define the rects.
        Press 'r' to reset, 's' to save and finish.

        Requires a display (won't work headless).
        """
        h, w = frame.shape[:2]
        calibrator = cls.from_defaults(w, h)
        points: list[tuple[int, int]] = []
        board_index = 0  # 0 = player, 1 = opponent
        board_names = ["PLAYER BOARD", "OPPONENT BOARD"]
        display = frame.copy()

        def _draw_overlay(img, cal: "GridCalibrator"):
            out = img.copy()
            for player in ("player", "opponent"):
                color = (0, 255, 0) if player == "player" else (255, 100, 0)
                for row in range(cal._data.rows):
                    for col in range(cal._data.cols):
                        x, y, cw, ch = cal._cell_rect(player, row, col)
                        cv2.rectangle(out, (x, y), (x + cw, y + ch), color, 1)
            return out

        def _on_mouse(event, x, y, _flags, _param):
            nonlocal board_index, display
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 2:
                    # Two points define a board rect
                    x0, y0 = points[0]
                    x1, y1 = points[1]
                    rect = GridRect(
                        x=min(x0, x1), y=min(y0, y1),
                        w=abs(x1 - x0), h=abs(y1 - y0)
                    )
                    if board_index == 0:
                        calibrator._data.player_board = rect
                    else:
                        calibrator._data.opponent_board = rect
                    points.clear()
                    board_index += 1
                    display = _draw_overlay(frame, calibrator)
                    if board_index >= 2:
                        print("Both boards calibrated. Press 's' to save, 'r' to reset.")

        win = "RRGA Grid Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _on_mouse)

        print(f"Click top-left then bottom-right corners of the {board_names[0]}.")
        while True:
            prompt = board_names[board_index] if board_index < 2 else "DONE — press s"
            overlay = _draw_overlay(display, calibrator)
            cv2.putText(overlay, f"Calibrating: {prompt}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s") and board_index >= 2:
                break
            elif key == ord("r"):
                board_index = 0
                points.clear()
                display = frame.copy()
                calibrator = cls.from_defaults(w, h)
                print(f"Reset. Click corners for {board_names[0]}.")
            elif key == 27:  # Esc
                break

        cv2.destroyWindow(win)
        calibrator._data.frame_width = w
        calibrator._data.frame_height = h
        if save_path:
            calibrator.save(save_path)
        return calibrator

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = asdict(self._data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        print(f"Calibration saved to {path}")

    # ------------------------------------------------------------------
    # Cell geometry
    # ------------------------------------------------------------------

    def _board_rect(self, player: str) -> GridRect:
        if player == "player":
            return self._data.player_board
        return self._data.opponent_board

    def _cell_rect(self, player: str, row: int, col: int) -> tuple[int, int, int, int]:
        """
        Returns (x, y, w, h) for the cell. Row 0 is the BACK row (visually:
        bottom of player board, top of opponent board).
        """
        board = self._board_rect(player)
        rows = self._data.rows
        # Player row 0 = back = visual bottom → invert row index for player
        # Opponent row 0 = back = visual top → no inversion needed
        if player == "player":
            visual_row = (rows - 1) - row
        else:
            visual_row = row
        return board.cell_rect(visual_row, col, rows, self._data.cols)

    def crop_cell(self, frame: np.ndarray, player: str,
                  row: int, col: int) -> np.ndarray:
        """
        Crops and returns the image for cell (row, col) on the given player's board.
        player: 'player' or 'opponent'
        row: 0 (back) – 4 (front)
        col: 0 (left) – 2 (right)
        """
        x, y, w, h = self._cell_rect(player, row, col)
        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        return frame[y1:y2, x1:x2].copy()

    def all_cell_crops(self, frame: np.ndarray
                       ) -> list[tuple[str, int, int, np.ndarray]]:
        """
        Returns crops for all 30 cells (both boards).
        Each item: (player, row, col, crop_image)
        """
        crops = []
        for player in ("player", "opponent"):
            for row in range(self._data.rows):
                for col in range(self._data.cols):
                    crops.append((player, row, col,
                                  self.crop_cell(frame, player, row, col)))
        return crops

    def crop_deck_icons(self, frame: np.ndarray,
                        player: str) -> list[np.ndarray]:
        """
        Crops and returns the 5 deck icon images for the given player.
        player: 'player' or 'opponent'

        The deck strip is divided into DECK_SIZE equal slots horizontally.
        Returns a list of DECK_SIZE BGR images, one per deck slot.
        """
        rect = (self._data.player_deck if player == "player"
                else self._data.opponent_deck)
        fh, fw = frame.shape[:2]
        slot_w = rect.w // DECK_SIZE
        icons = []
        for i in range(DECK_SIZE):
            x1 = max(0, rect.x + i * slot_w)
            y1 = max(0, rect.y)
            x2 = min(fw, x1 + slot_w)
            y2 = min(fh, rect.y + rect.h)
            icons.append(frame[y1:y2, x1:x2].copy())
        return icons

    def draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Returns a copy of frame with both board grids drawn on it."""
        out = frame.copy()
        for player in ("player", "opponent"):
            color = (0, 220, 0) if player == "player" else (220, 100, 0)
            for row in range(self._data.rows):
                for col in range(self._data.cols):
                    x, y, w, h = self._cell_rect(player, row, col)
                    cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(out, f"{row},{col}", (x + 2, y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._data.frame_width, self._data.frame_height

    @property
    def cell_size(self) -> tuple[int, int]:
        """(width, height) of a single cell in pixels."""
        b = self._data.player_board
        return b.w // self._data.cols, b.h // self._data.rows
