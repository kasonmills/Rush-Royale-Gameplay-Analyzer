"""
Main application window — Rush Royale Gameplay Analyzer.

Threading model:
    Main thread  — Qt event loop + all UI widgets
    Worker thread — MatchRunner.run() (blocking for match duration)
    Bridge: MatchWorker emits Qt signals from the worker thread;
            MainWindow slots receive them safely on the main thread.

Graceful degradation:
    - Missing sprite library → boards show unit_id text instead of images
    - Missing calibration    → defaults are used automatically
    - Missing DB             → databases are created on first Start
    - Phase 3 stubs active   → win probability uses Phase 1/2 components only
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup, QLineEdit,
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QHeaderView, QFrame, QStatusBar,
    QSizePolicy, QTabWidget, QScrollArea,
)
from PyQt6.QtGui import QFont, QPalette, QColor

from src.analysis.game_state import GameState, BoardState
from src.analysis.match_runner import MatchRunner, MatchResult, MatchRunnerConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BG           = "#1e1e1e"
_PANEL        = "#2a2a2a"
_CELL_EMPTY   = "#303030"
_CELL_OCC     = "#1c3d5a"
_CELL_BORDER  = "#5599cc"
_TEXT         = "#d8d8d8"
_HERO_COLOR   = "#ffcc44"

_APP_STYLE = f"""
QWidget           {{ background-color: {_BG}; color: {_TEXT}; font-family: "Segoe UI", Arial, sans-serif; }}
QGroupBox         {{ border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 6px; }}
QGroupBox::title  {{ subcontrol-origin: margin; left: 8px; color: #aaa; }}
QLineEdit         {{ background-color: #2d2d2d; border: 1px solid #555; border-radius: 3px;
                     padding: 3px 6px; color: {_TEXT}; }}
QPushButton       {{ background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
                     padding: 5px 12px; color: {_TEXT}; }}
QPushButton:hover {{ background-color: #4a4a4a; }}
QPushButton:disabled {{ color: #666; border-color: #3a3a3a; }}
QRadioButton               {{ spacing: 6px; color: #bbb; padding: 3px 6px;
                              border-radius: 4px; }}
QRadioButton:hover         {{ background-color: #353535; color: {_TEXT}; }}
QRadioButton:checked       {{ color: #5599cc; }}
QRadioButton::indicator              {{ width: 13px; height: 13px; }}
QTableWidget      {{ background-color: #252525; gridline-color: #3a3a3a;
                     alternate-background-color: #2c2c2c; }}
QHeaderView::section {{ background-color: #333; border: none; padding: 4px; }}
QStatusBar        {{ background-color: #181818; color: #aaa; }}
QTabBar::tab               {{ background-color: #252525; color: #999;
                              padding: 6px 18px; border: 1px solid #3a3a3a;
                              border-bottom: none; border-radius: 4px 4px 0 0; }}
QTabBar::tab:selected      {{ background-color: {_PANEL}; color: {_TEXT};
                              border-color: #555; }}
QTabBar::tab:hover:!selected {{ background-color: #303030; color: #ccc; }}
QTabWidget::pane           {{ border: 1px solid #444; background-color: {_PANEL}; }}
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_unit_names() -> dict[str, str]:
    """Return unit_id → display_name from unit_meta.db (empty dict on failure)."""
    try:
        from src.database.connection import unit_meta_db
        with unit_meta_db() as conn:
            rows = conn.execute(
                "SELECT unit_id, display_name FROM units"
            ).fetchall()
            return {r["unit_id"]: r["display_name"] for r in rows}
    except Exception:
        return {}


def _load_recent_matches(limit: int = 20) -> list[sqlite3.Row]:
    """Return the most recent matches from match_history.db."""
    try:
        from src.database.connection import _DB_PATHS
        db_path = Path(_DB_PATHS["match_history"])
        if not db_path.exists():
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT match_id, recorded_at, source_type,
                      player_hero_id, total_waves, outcome
               FROM matches
               ORDER BY recorded_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _save_outcome(match_id: str, outcome: str):
    from src.database.connection import _DB_PATHS
    from src.database.match_history_repo import MatchRepo
    conn = sqlite3.connect(_DB_PATHS["match_history"])
    MatchRepo.set_outcome(conn, match_id, outcome)
    conn.commit()
    conn.close()


def _purge_empty_matches() -> int:
    """
    Remove matches that recorded zero snapshots (pipeline opened but saw nothing).
    Returns the count deleted, or 0 if the DB doesn't exist yet.
    """
    try:
        from src.database.connection import _DB_PATHS
        from src.database.match_history_repo import MatchRepo
        db_path = Path(_DB_PATHS["match_history"])
        if not db_path.exists():
            return 0
        conn = sqlite3.connect(db_path)
        deleted = MatchRepo.purge_empty(conn)
        conn.commit()
        conn.close()
        return deleted
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Unit card helpers
# ---------------------------------------------------------------------------

# Per-unit icon background colours — derived by hashing unit_id so each unit
# gets a stable colour without needing real sprites.
_ICON_PALETTE = [
    "#1c3d5a", "#3d1c5a", "#1c5a3d", "#5a3d1c",
    "#5a1c3d", "#3d5a1c", "#1c4a4a", "#4a1c1c",
    "#2a4a2a", "#4a2a4a", "#2a2a5a", "#5a4a1c",
]


def _icon_color(unit_id: str) -> str:
    return _ICON_PALETTE[hash(unit_id) % len(_ICON_PALETTE)]


def _initials(display_name: str) -> str:
    """Up to 2 uppercase initials from the display name."""
    parts = display_name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return display_name[:2].upper()


def _talent_label(unit_cell) -> str:
    """Return a short talent string, e.g. 'T2·R', 'T1·?', or '—'."""
    tier = unit_cell.highest_talent_tier
    if tier is None:
        return "—"
    branch = unit_cell.talent_path.get(tier)
    if branch is None:
        return f"T{tier}·?"
    return f"T{tier}·{branch[0]}"   # L / R / F


# ---------------------------------------------------------------------------
# UnitCardWidget — compact card for one unit
# ---------------------------------------------------------------------------

class UnitCardWidget(QFrame):
    """
    Shows a single unit: coloured icon placeholder, rank badge,
    talent badge, and truncated name.

    Rank is prefixed with '~' and coloured amber when recognition
    confidence is below 0.70 (i.e. the value is estimated, not confirmed).
    """

    _CARD_W = 72
    _CARD_H = 88
    _CONF_THRESHOLD = 0.70

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._CARD_W, self._CARD_H)
        self.setStyleSheet(
            "background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # Icon placeholder (coloured square with initials)
        self._icon = QLabel()
        self._icon.setFixedSize(40, 40)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._icon.setStyleSheet(
            "background-color: #1c3d5a; border-radius: 4px; color: #ddd; border: none;"
        )
        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.addStretch()
        icon_row.addWidget(self._icon)
        icon_row.addStretch()
        root.addLayout(icon_row)

        # Rank + talent on one line
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(4)
        self._rank_lbl = QLabel()
        self._rank_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        self._talent_lbl = QLabel()
        self._talent_lbl.setFont(QFont("Segoe UI", 7))
        self._talent_lbl.setStyleSheet("color: #aaa; border: none;")
        badge_row.addStretch()
        badge_row.addWidget(self._rank_lbl)
        badge_row.addWidget(self._talent_lbl)
        badge_row.addStretch()
        root.addLayout(badge_row)

        # Unit name (elided)
        self._name_lbl = QLabel()
        self._name_lbl.setFont(QFont("Segoe UI", 6))
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setStyleSheet("color: #aaa; border: none;")
        self._name_lbl.setMaximumWidth(self._CARD_W - 8)
        root.addWidget(self._name_lbl)

    def set_unit(self, display_name: str, unit_cell):
        conf  = unit_cell.recognition_confidence
        rank  = unit_cell.merge_rank
        known = conf >= self._CONF_THRESHOLD

        color = _icon_color(unit_cell.unit_id)
        self._icon.setStyleSheet(
            f"background-color: {color}; border-radius: 4px; color: #ddd; border: none;"
        )
        self._icon.setText(_initials(display_name))

        if known:
            rank_text  = f"R{rank}"
            rank_style = f"color: {_TEXT}; border: none;"
        else:
            rank_text  = f"~R{rank}"
            rank_style = "color: #d4a017; border: none;"   # amber = estimated

        self._rank_lbl.setText(rank_text)
        self._rank_lbl.setStyleSheet(rank_style)
        self._rank_lbl.setToolTip(
            f"Confidence: {conf:.0%}" + ("" if known else " (estimated)")
        )

        self._talent_lbl.setText(_talent_label(unit_cell))
        self._name_lbl.setText(display_name)
        self.setToolTip(
            f"{display_name}\nRank {rank}  conf={conf:.0%}\n{_talent_label(unit_cell)}"
        )
        self.show()

    def clear(self):
        self.hide()


# ---------------------------------------------------------------------------
# UnitRosterWidget — horizontal scrolling strip of UnitCardWidgets
# ---------------------------------------------------------------------------

_MAX_ROSTER_SLOTS = 15   # 3 cols × 5 rows = max 15 units per board


class UnitRosterWidget(QGroupBox):
    """Horizontal scrolling strip showing all units for one side."""

    def __init__(self, title: str, unit_names: dict[str, str], parent=None):
        super().__init__(title, parent)
        self._unit_names = unit_names

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 8, 4, 4)
        outer.setSpacing(3)

        self._hero_lbl = QLabel("Hero: —")
        self._hero_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._hero_lbl.setStyleSheet(f"color: {_HERO_COLOR}; border: none;")
        self._hero_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._hero_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(UnitCardWidget._CARD_H + 16)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        self._cards_layout = QHBoxLayout(inner)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(5)
        self._cards_layout.addStretch()

        self._cards: list[UnitCardWidget] = []
        for _ in range(_MAX_ROSTER_SLOTS):
            card = UnitCardWidget(inner)
            card.hide()
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            self._cards.append(card)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def update_hero(self, hero_id: Optional[str]):
        name = (self._unit_names.get(hero_id, hero_id) if hero_id else "—")
        self._hero_lbl.setText(f"Hero: {name}")

    def update_roster(self, board: "BoardState"):
        units = board.occupied()          # list of (row, col, UnitCell)
        for i, card in enumerate(self._cards):
            if i < len(units):
                _, _, uc = units[i]
                name = self._unit_names.get(uc.unit_id, uc.unit_id)
                card.set_unit(name, uc)
            else:
                card.clear()

    def clear_roster(self):
        self._hero_lbl.setText("Hero: —")
        for card in self._cards:
            card.clear()


# ---------------------------------------------------------------------------
# CellWidget — single board cell
# ---------------------------------------------------------------------------

class CellWidget(QFrame):
    """Displays one 3×5 board cell. Shows unit name + rank when occupied."""

    _EMPTY_STYLE = (
        f"background-color: {_CELL_EMPTY}; border: 1px solid #444; border-radius: 2px;"
    )
    _OCC_STYLE = (
        f"background-color: {_CELL_OCC}; border: 1px solid {_CELL_BORDER}; border-radius: 2px;"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(82, 58)
        self.setStyleSheet(self._EMPTY_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(1)

        self._name = QLabel("", self)
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        self._name.setStyleSheet(f"color: {_TEXT}; background: transparent; border: none;")
        self._name.setWordWrap(True)

        self._sub = QLabel("", self)
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub.setFont(QFont("Segoe UI", 6))
        self._sub.setStyleSheet("color: #999; background: transparent; border: none;")

        layout.addWidget(self._name)
        layout.addWidget(self._sub)

    def set_unit(self, display_name: str, rank: int, talent_tier: Optional[int]):
        self._name.setText(display_name)
        sub = f"R{rank}"
        if talent_tier:
            sub += f"  T{talent_tier}"
        self._sub.setText(sub)
        self.setStyleSheet(self._OCC_STYLE)

    def clear(self):
        self._name.setText("")
        self._sub.setText("")
        self.setStyleSheet(self._EMPTY_STYLE)


# ---------------------------------------------------------------------------
# BoardWidget — 5×3 grid of CellWidgets
# ---------------------------------------------------------------------------

class BoardWidget(QGroupBox):
    """One player's full board: 5 rows × 3 columns of CellWidgets."""

    def __init__(self, title: str, unit_names: dict[str, str], parent=None):
        super().__init__(title, parent)
        self._unit_names = unit_names

        outer = QVBoxLayout(self)
        outer.setSpacing(4)

        # Display: 3 rows (one per board column/lane) × 5 columns (depth positions)
        # Mapping from BoardState (game_row 0-4, game_col 0-2):
        #   visual_row = game_col  (0, 1, 2)
        #   visual_col = game_row  (0, 1, 2, 3, 4)
        grid = QGridLayout()
        grid.setSpacing(3)
        self._cells: list[list[CellWidget]] = []   # [visual_row][visual_col]
        for vrow in range(3):
            row_cells = []
            for vcol in range(5):
                cell = CellWidget(self)
                grid.addWidget(cell, vrow, vcol)
                row_cells.append(cell)
            self._cells.append(row_cells)
        outer.addLayout(grid)

    def update_board(self, board: BoardState):
        # Clear all first
        for row in self._cells:
            for cell in row:
                cell.clear()
        # Fill occupied — map (game_row, game_col) → (visual_row=game_col, visual_col=game_row)
        for game_row, game_col, unit_cell in board.occupied():
            name = self._unit_names.get(unit_cell.unit_id, unit_cell.unit_id)
            self._cells[game_col][game_row].set_unit(
                name, unit_cell.merge_rank, unit_cell.highest_talent_tier
            )

    def clear_all(self):
        for row in self._cells:
            for cell in row:
                cell.clear()


# ---------------------------------------------------------------------------
# MatchWorker — wraps MatchRunner for background execution
# ---------------------------------------------------------------------------

class MatchWorker(QObject):
    """
    Runs MatchRunner.run() in a QThread and emits Qt signals on each frame.

    Usage:
        worker = MatchWorker()
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(lambda: worker.start_match("video", "/path/file.mp4"))
        worker.state_updated.connect(main_window.on_state)
        worker.match_finished.connect(main_window.on_finished)
        thread.start()
    """

    state_updated  = pyqtSignal(object)   # GameState
    match_finished = pyqtSignal(object)   # MatchResult
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._runner: Optional[MatchRunner] = None

    def start_match(self, source_type: str, path: str = ""):
        """Build and run the MatchRunner — blocks until the match ends."""
        try:
            cfg = MatchRunnerConfig(persist=True)

            if source_type == "video":
                self._runner = MatchRunner.for_video(path, config=cfg)
            elif source_type == "window":
                self._runner = MatchRunner.for_window(config=cfg)
            else:
                self._runner = MatchRunner.for_scrcpy(config=cfg)

            result = self._runner.run(on_state=self.state_updated.emit)
            self.match_finished.emit(result)

        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def stop(self):
        """Signal the runner to stop after the current frame."""
        if self._runner is not None:
            self._runner.stop()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rush Royale Gameplay Analyzer")
        self.setMinimumSize(920, 720)

        self._unit_names  = _load_unit_names()
        self._worker:     Optional[MatchWorker] = None
        self._thread:     Optional[QThread]     = None
        self._last_result: Optional[MatchResult] = None

        self._build_ui()
        self._refresh_history()

        # Refresh history periodically while idle
        self._hist_timer = QTimer(self)
        self._hist_timer.setInterval(15_000)
        self._hist_timer.timeout.connect(self._refresh_history)
        self._hist_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._build_analyzer_tab(), "🎮  Analyzer")
        self._tabs.addTab(self._build_history_tab(),  "📋  Match History")
        root.addWidget(self._tabs)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._set_status("Idle — select a source and press Start.")

    def _build_analyzer_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 4)

        layout.addWidget(self._build_source_panel())

        # Board view — collapsible 3×5 grids (toggled by button in source panel)
        self._board_container = QWidget()
        boards_row = QHBoxLayout(self._board_container)
        boards_row.setSpacing(8)
        boards_row.setContentsMargins(0, 0, 0, 0)
        self._player_board = BoardWidget("Player",   self._unit_names, self)
        self._opp_board    = BoardWidget("Opponent", self._unit_names, self)
        boards_row.addWidget(self._player_board)
        boards_row.addWidget(self._opp_board)
        self._board_container.setVisible(False)   # hidden by default
        layout.addWidget(self._board_container)

        layout.addWidget(self._build_hud_panel())
        layout.addWidget(self._build_prob_panel())
        layout.addWidget(self._build_composition_panel())
        layout.addStretch()
        return tab

    def _build_source_panel(self) -> QGroupBox:
        box = QGroupBox("Source")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # Mode radios
        radio_row = QHBoxLayout()
        self._btn_group = QButtonGroup(self)
        self._rb_video  = QRadioButton("Video File")
        self._rb_window = QRadioButton("Window Capture")
        self._rb_scrcpy = QRadioButton("scrcpy (Android)")
        self._rb_video.setChecked(True)
        for rb in (self._rb_video, self._rb_window, self._rb_scrcpy):
            self._btn_group.addButton(rb)
            radio_row.addWidget(rb)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # Video path row
        path_row = QHBoxLayout()
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Path to video file…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_video)
        path_row.addWidget(self._path_input)
        path_row.addWidget(browse_btn)
        self._path_row_widget = QWidget()
        self._path_row_widget.setLayout(path_row)
        layout.addWidget(self._path_row_widget)

        # Toggle path row visibility
        self._rb_video.toggled.connect(self._path_row_widget.setVisible)
        for rb in (self._rb_window, self._rb_scrcpy):
            rb.toggled.connect(
                lambda checked, w=self._path_row_widget: w.setVisible(not checked)
            )

        # Start / Stop / Board toggle
        ctrl_row = QHBoxLayout()
        self._toggle_board_btn = QPushButton("⊞  Show Board")
        self._toggle_board_btn.setFixedWidth(140)
        self._toggle_board_btn.setCheckable(True)
        self._toggle_board_btn.setChecked(False)
        self._toggle_board_btn.clicked.connect(self._toggle_board)
        ctrl_row.addWidget(self._toggle_board_btn)
        ctrl_row.addStretch()
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedWidth(130)
        self._stop_btn  = QPushButton("■  Stop")
        self._stop_btn.setFixedWidth(130)
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start_analysis)
        self._stop_btn.clicked.connect(self._stop_analysis)
        ctrl_row.addWidget(self._start_btn)
        ctrl_row.addWidget(self._stop_btn)
        layout.addLayout(ctrl_row)

        return box

    def _build_hud_panel(self) -> QGroupBox:
        box = QGroupBox("Match HUD")
        row = QHBoxLayout(box)
        row.setSpacing(0)

        font = QFont("Segoe UI", 9)

        def _hud(label):
            lbl = QLabel(f"{label}: —")
            lbl.setFont(font)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(lbl)
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet("color: #444;")
            row.addWidget(sep)
            return lbl

        self._wave_lbl  = _hud("Wave")
        self._php_lbl   = _hud("Player HP")
        self._ohp_lbl   = _hud("Opp HP")
        self._mana_lbl  = _hud("Mana")
        self._conf_lbl  = _hud("Confidence")
        self._frame_lbl = _hud("Frames")
        return box

    def _build_prob_panel(self) -> QGroupBox:
        box = QGroupBox("Win Probability")
        layout = QVBoxLayout(box)
        self._prob_bar = QProgressBar()
        self._prob_bar.setRange(0, 100)
        self._prob_bar.setValue(50)
        self._prob_bar.setMinimumHeight(30)
        self._update_prob_bar(0.5)
        layout.addWidget(self._prob_bar)
        return box

    def _build_composition_panel(self) -> QGroupBox:
        box = QGroupBox("Unit Composition")
        col = QVBoxLayout(box)
        col.setSpacing(4)
        col.setContentsMargins(4, 8, 4, 4)

        # Legend row
        legend = QHBoxLayout()
        for text, style in [
            ("■  Icon colour = unit type (placeholder until sprites load)",
             "color: #777; font-size: 9px;"),
            ("  ~R# = estimated rank (low confidence)",
             "color: #d4a017; font-size: 9px;"),
            ("  T#·L/R/F = talent tier · branch",
             "color: #aaa; font-size: 9px;"),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(style)
            legend.addWidget(lbl)
        legend.addStretch()
        col.addLayout(legend)

        # Two rosters side by side
        rosters_row = QHBoxLayout()
        rosters_row.setSpacing(6)
        self._player_roster = UnitRosterWidget("Player Units", self._unit_names, box)
        self._opp_roster    = UnitRosterWidget("Opponent Units", self._unit_names, box)
        rosters_row.addWidget(self._player_roster)
        rosters_row.addWidget(self._opp_roster)
        col.addLayout(rosters_row)
        return box

    def _build_history_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        heading = QLabel("Last 20 matches — select a row to tag its outcome.")
        heading.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(heading)

        self._hist_table = QTableWidget(0, 6)
        self._hist_table.setHorizontalHeaderLabels(
            ["Match ID", "Recorded", "Source", "Player Hero", "Waves", "Outcome"]
        )
        hh = self._hist_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._hist_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._hist_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._hist_table.setAlternatingRowColors(True)
        layout.addWidget(self._hist_table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._win_btn  = QPushButton("✓  Mark Win")
        self._loss_btn = QPushButton("✗  Mark Loss")
        for btn in (self._win_btn, self._loss_btn):
            btn.setFixedWidth(130)
            btn.setEnabled(False)
        self._win_btn.clicked.connect(lambda: self._set_outcome("win"))
        self._loss_btn.clicked.connect(lambda: self._set_outcome("loss"))
        btn_row.addWidget(self._win_btn)
        btn_row.addWidget(self._loss_btn)
        layout.addLayout(btn_row)

        return tab

    # ------------------------------------------------------------------
    # Analysis control
    # ------------------------------------------------------------------

    def _toggle_board(self, checked: bool):
        self._board_container.setVisible(checked)
        self._toggle_board_btn.setText("⊟  Hide Board" if checked else "⊞  Show Board")
        # Shrink window back down when boards are hidden
        if not checked:
            self.adjustSize()

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", str(Path.home()),
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
        )
        if path:
            self._path_input.setText(path)

    def _start_analysis(self):
        if self._rb_video.isChecked():
            source_type = "video"
            path = self._path_input.text().strip()
            if not path:
                self._set_status("Please select a video file first.")
                return
            if not Path(path).exists():
                self._set_status(f"File not found: {path}")
                return
        elif self._rb_window.isChecked():
            source_type = "window"
            path = ""
        else:
            source_type = "scrcpy"
            path = ""

        # Spin up worker thread
        self._worker = MatchWorker()
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(
            lambda: self._worker.start_match(source_type, path)
        )
        self._worker.state_updated.connect(self._on_state)
        self._worker.match_finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)

        self._thread.start()

        # UI state — running
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._win_btn.setEnabled(False)
        self._loss_btn.setEnabled(False)
        self._player_board.clear_all()
        self._opp_board.clear_all()
        self._player_roster.clear_roster()
        self._opp_roster.clear_roster()
        self._update_prob_bar(0.5)
        self._set_status("Running…  (no sprites = board shows unit IDs; recognition stubs active)")

    def _stop_analysis(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)
        self._set_status("Stop requested — finishing current frame…")

    # ------------------------------------------------------------------
    # Signal handlers (always called on the main thread)
    # ------------------------------------------------------------------

    def _on_state(self, state: GameState):
        self._player_board.update_board(state.player_board)
        self._opp_board.update_board(state.opponent_board)
        self._player_roster.update_hero(state.player_hero_id)
        self._opp_roster.update_hero(state.opponent_hero_id)
        self._player_roster.update_roster(state.player_board)
        self._opp_roster.update_roster(state.opponent_board)

        self._wave_lbl.setText(f"Wave: {state.wave_number or '—'}")
        self._php_lbl.setText(f"Player HP: {state.player_hp if state.player_hp is not None else '—'}")
        self._ohp_lbl.setText(f"Opp HP: {state.opponent_hp if state.opponent_hp is not None else '—'}")
        self._mana_lbl.setText(f"Mana: {state.player_mana if state.player_mana is not None else '—'}")
        self._conf_lbl.setText(f"Confidence: {state.pipeline_confidence:.0%}")
        self._frame_lbl.setText(f"Frames: {state.source_frame_index or 0}")

        if state.win_probability is not None:
            self._update_prob_bar(state.win_probability)

    def _on_finished(self, result: MatchResult):
        self._last_result = result
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        if result.total_snapshots_written == 0:
            # Nothing was observed — delete the ghost match record immediately
            _purge_empty_matches()
            self._last_result = None
            self._win_btn.setEnabled(False)
            self._loss_btn.setEnabled(False)
            self._set_status(
                "Session ended — no frames were captured, so no match was recorded."
            )
        else:
            self._win_btn.setEnabled(True)
            self._loss_btn.setEnabled(True)
            self._set_status(
                f"Match complete — {result.total_frames_processed} frames in "
                f"{result.duration_sec:.1f}s.  Tag the outcome below."
            )
        self._refresh_history()

    def _on_error(self, message: str):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_status(f"Error: {message}")

    # ------------------------------------------------------------------
    # Outcome tagging
    # ------------------------------------------------------------------

    def _set_outcome(self, outcome: str):
        if self._last_result is None:
            return
        try:
            _save_outcome(self._last_result.match_id, outcome)
            self._win_btn.setEnabled(False)
            self._loss_btn.setEnabled(False)
            self._set_status(f"Outcome '{outcome}' saved for match {self._last_result.match_id[:8]}…")
            self._refresh_history()
        except Exception as exc:
            self._set_status(f"Could not save outcome: {exc}")

    # ------------------------------------------------------------------
    # Match history table
    # ------------------------------------------------------------------

    def _refresh_history(self):
        _purge_empty_matches()
        rows = _load_recent_matches()
        self._hist_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            short_id  = str(row["match_id"])[:8] + "…"
            recorded  = str(row["recorded_at"] or "")[:16].replace("T", " ")
            source    = str(row["source_type"] or "")
            hero      = (self._unit_names.get(row["player_hero_id"] or "")
                         or row["player_hero_id"] or "—")
            waves     = str(row["total_waves"] or "—")
            outcome   = str(row["outcome"] or "—")

            for j, val in enumerate([short_id, recorded, source, hero, waves, outcome]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 5 and val == "win":
                    item.setForeground(QColor("#5dba7d"))
                elif j == 5 and val == "loss":
                    item.setForeground(QColor("#ba5d5d"))
                self._hist_table.setItem(i, j, item)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_prob_bar(self, prob: float):
        pct = int(prob * 100)
        if pct >= 65:
            label = "Ahead"
        elif pct <= 35:
            label = "Behind"
        else:
            label = "Even"

        # Smooth hue: 0 % → red (hue 0°), 100 % → green (hue 120°).
        # Qt fromHsvF expects hue in [0, 1] where 1 = 360°, so divide by 3.
        hue = prob / 3.0
        color = QColor.fromHsvF(hue, 0.85, 0.78).name()

        self._prob_bar.setValue(pct)
        self._prob_bar.setFormat(f"{pct}%  ({label})")
        self._prob_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 11px;
                background-color: #252525;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)

    def _set_status(self, message: str):
        self._status_bar.showMessage(message)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def run():
    """Launch the RRGA desktop application."""
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(_APP_STYLE)

    # Use system dark palette as base
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(_BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#252525"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#2c2c2c"))
    palette.setColor(QPalette.ColorRole.Text,            QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#3a3a3a"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#1c5a8a"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())