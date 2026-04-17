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
    _CARD_H = 104
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

        # Summon percentage label — sits above the icon
        self._pct_lbl = QLabel("—")
        self._pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pct_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._pct_lbl.setStyleSheet("color: #666; border: none;")
        root.addWidget(self._pct_lbl)

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

    def set_pct(self, observed: int, total: int, expected_rate: float):
        """
        Update the summon-frequency percentage label above the icon.

        Colour coding vs expected rate:
          green  — observed > expected * 1.15  (summoning more than expected)
          red    — observed < expected * 0.85  (summoning less than expected)
          white  — within ±15% of expected
          grey   — no data (total == 0)
        """
        if total == 0:
            self._pct_lbl.setText("—")
            self._pct_lbl.setStyleSheet("color: #666; border: none;")
            return

        obs_rate = observed / total
        pct_text = f"{obs_rate:.0%}"
        if obs_rate > expected_rate * 1.15:
            color = "#5dba7d"   # green — above expected
        elif obs_rate < expected_rate * 0.85:
            color = "#ba5d5d"   # red — below expected
        else:
            color = _TEXT       # white — roughly on target
        self._pct_lbl.setText(pct_text)
        self._pct_lbl.setStyleSheet(f"color: {color}; border: none;")
        self._pct_lbl.setToolTip(
            f"Summons: {observed}/{total}  ({obs_rate:.1%} observed vs "
            f"{expected_rate:.1%} expected)"
        )

    def clear_pct(self):
        self._pct_lbl.setText("—")
        self._pct_lbl.setStyleSheet("color: #666; border: none;")

    def clear(self):
        self.clear_pct()
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
        scroll.setFixedHeight(UnitCardWidget._CARD_H + 18)
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

    def update_roster(self, board: "BoardState",
                      summon_counts: Optional[dict] = None):
        units = board.occupied()          # list of (row, col, UnitCell)
        total = sum(summon_counts.values()) if summon_counts else 0
        deck_size = len(summon_counts) if summon_counts else 0
        expected_rate = 1.0 / deck_size if deck_size > 0 else 0.0

        for i, card in enumerate(self._cards):
            if i < len(units):
                _, _, uc = units[i]
                name = self._unit_names.get(uc.unit_id, uc.unit_id)
                card.set_unit(name, uc)
                if summon_counts is not None and total > 0:
                    card.set_pct(
                        summon_counts.get(uc.unit_id, 0),
                        total,
                        expected_rate,
                    )
                else:
                    card.clear_pct()
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

        # Live summon tracking (UI-side, for the percentage labels)
        from src.analysis.game_state import BoardState as _BS
        self._player_prev_board: Optional[_BS] = None
        self._opp_prev_board:    Optional[_BS] = None
        self._player_summon_counts: dict[str, int] = {}
        self._opp_summon_counts:    dict[str, int] = {}

        self._build_ui()
        self._refresh_history()
        self._show_placeholder_state()

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
        self._tabs.addTab(self._build_analyzer_tab(),  "🎮  Analyzer")
        self._tabs.addTab(self._build_history_tab(),   "📋  Match History")
        self._tabs.addTab(self._build_summon_tab(),    "🎲  Summon Stats")
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
        self._hist_table.itemSelectionChanged.connect(self._on_history_selected)
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

        # Summon breakdown panel — shown when a row is selected
        self._hist_detail_box = QGroupBox("Summon Breakdown")
        detail_layout = QVBoxLayout(self._hist_detail_box)
        self._hist_detail_table = QTableWidget(0, 4)
        self._hist_detail_table.setHorizontalHeaderLabels(
            ["Unit", "Summons", "Rate", "vs Expected"]
        )
        dh = self._hist_detail_table.horizontalHeader()
        dh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._hist_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._hist_detail_table.setAlternatingRowColors(True)
        self._hist_detail_table.setMaximumHeight(160)
        detail_layout.addWidget(self._hist_detail_table)
        self._hist_detail_box.setVisible(False)
        layout.addWidget(self._hist_detail_box)

        return tab

    def _build_summon_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # ---- Controls row ----
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Context:"))
        self._summon_ctx_group = QButtonGroup(self)
        for label, value in [("All", "all"), ("Manual only", "manual"),
                              ("Post-merge only", "post_merge")]:
            rb = QRadioButton(label)
            rb.setProperty("ctx_value", value)
            if value == "all":
                rb.setChecked(True)
            self._summon_ctx_group.addButton(rb)
            ctrl.addWidget(rb)
        ctrl.addStretch()
        self._summon_refresh_btn = QPushButton("↻  Refresh")
        self._summon_refresh_btn.setFixedWidth(110)
        self._summon_refresh_btn.clicked.connect(self._refresh_summon_stats)
        ctrl.addWidget(self._summon_refresh_btn)
        layout.addLayout(ctrl)

        # ---- Verdict banner ----
        self._summon_verdict = QLabel("No data yet")
        self._summon_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summon_verdict.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._summon_verdict.setStyleSheet(
            "color: #aaa; border: 1px solid #444; border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(self._summon_verdict)

        # ---- Chi-squared summary ----
        self._chi_lbl = QLabel("")
        self._chi_lbl.setStyleSheet("color: #888; font-size: 9px;")
        self._chi_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._chi_lbl)

        # ---- Per-unit stats table ----
        unit_box = QGroupBox("Per-Unit Summon Distribution")
        unit_layout = QVBoxLayout(unit_box)
        self._summon_unit_table = QTableWidget(0, 8)
        self._summon_unit_table.setHorizontalHeaderLabels([
            "Unit", "Observed", "Expected", "Obs %", "Exp %",
            "Z-score", "p-value", "95% CI",
        ])
        hh = self._summon_unit_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._summon_unit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._summon_unit_table.setAlternatingRowColors(True)
        self._summon_unit_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        unit_layout.addWidget(self._summon_unit_table)
        layout.addWidget(unit_box)

        # ---- Merge breakdown table ----
        merge_box = QGroupBox("Merge Event Breakdown")
        merge_layout = QVBoxLayout(merge_box)
        note = QLabel(
            "Tracks every detected merge. Used to verify whether post-merge "
            "summon pools follow the same distribution as manual summons."
        )
        note.setStyleSheet("color: #777; font-size: 9px;")
        note.setWordWrap(True)
        merge_layout.addWidget(note)
        self._summon_merge_table = QTableWidget(0, 4)
        self._summon_merge_table.setHorizontalHeaderLabels(
            ["Unit", "From Rank", "To Rank", "Count"]
        )
        mh = self._summon_merge_table.horizontalHeader()
        mh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._summon_merge_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._summon_merge_table.setAlternatingRowColors(True)
        merge_layout.addWidget(self._summon_merge_table)
        layout.addWidget(merge_box)

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
        self._player_summon_counts = {}
        self._opp_summon_counts    = {}
        self._player_prev_board    = None
        self._opp_prev_board       = None
        self._update_prob_bar(0.5)
        # Clear HUD placeholders
        for lbl in (self._wave_lbl, self._php_lbl, self._ohp_lbl,
                    self._mana_lbl, self._conf_lbl, self._frame_lbl):
            lbl.setText(lbl.text().split(":")[0] + ": —")
        self._hist_detail_box.setVisible(False)
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
        from src.analysis.match_runner import _detect_summons

        # Update live summon counts for both sides
        if self._player_prev_board is not None:
            for uid in _detect_summons(self._player_prev_board, state.player_board):
                self._player_summon_counts[uid] = \
                    self._player_summon_counts.get(uid, 0) + 1
        if self._opp_prev_board is not None:
            for uid in _detect_summons(self._opp_prev_board, state.opponent_board):
                self._opp_summon_counts[uid] = \
                    self._opp_summon_counts.get(uid, 0) + 1
        self._player_prev_board = state.player_board
        self._opp_prev_board    = state.opponent_board

        self._player_board.update_board(state.player_board)
        self._opp_board.update_board(state.opponent_board)
        self._player_roster.update_hero(state.player_hero_id)
        self._opp_roster.update_hero(state.opponent_hero_id)
        self._player_roster.update_roster(state.player_board, self._player_summon_counts)
        self._opp_roster.update_roster(state.opponent_board, self._opp_summon_counts)

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

        _END_REASON_LABELS = {
            "video_ended":       "Video finished.",
            "user_stopped":      "Stopped by user.",
            "idle_empty_board":  "Auto-stopped — board was empty for 12 s "
                                 "(results/lobby screen detected).",
            "idle_no_activity":  "Auto-stopped — no board changes, hero abilities, "
                                 "wave progress, or HP changes for 60 s "
                                 "(match appears to have ended).",
            "match_end_hp":      "Auto-stopped — a player's HP reached 0.",
        }
        reason_text = _END_REASON_LABELS.get(result.end_reason, result.end_reason)

        if result.total_snapshots_written == 0:
            # Nothing was observed — delete the ghost match record immediately
            _purge_empty_matches()
            self._last_result = None
            self._win_btn.setEnabled(False)
            self._loss_btn.setEnabled(False)
            self._set_status(
                f"Session ended — no frames captured, match not recorded.  "
                f"({reason_text})"
            )
        else:
            self._win_btn.setEnabled(True)
            self._loss_btn.setEnabled(True)
            self._set_status(
                f"Match complete — {result.total_frames_processed} frames, "
                f"{result.duration_sec:.1f}s.  {reason_text}  Tag the outcome below."
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

    def _refresh_summon_stats(self):
        try:
            from src.database.connection import _DB_PATHS
            from src.analysis.summon_analyzer import SummonAnalyzer
            import sqlite3 as _sq

            db_path = Path(_DB_PATHS["summon_analysis"])
            if not db_path.exists():
                self._summon_verdict.setText("Database not initialised yet — run a match first.")
                return

            # Determine selected context
            selected = next(
                (b for b in self._summon_ctx_group.buttons() if b.isChecked()), None
            )
            ctx = selected.property("ctx_value") if selected else "all"

            conn = _sq.connect(db_path)
            conn.row_factory = _sq.Row
            result = SummonAnalyzer.analyse(conn, trigger_type=ctx,
                                            unit_names=self._unit_names)
            conn.close()

            # Verdict banner colour
            if "Bias" in result.verdict:
                color = "#ba5d5d"
            elif "Suspicious" in result.verdict:
                color = "#d4a017"
            elif "Fair" in result.verdict:
                color = "#5dba7d"
            else:
                color = "#aaa"
            self._summon_verdict.setText(
                f"{result.verdict}  —  {result.verdict_detail}"
            )
            self._summon_verdict.setStyleSheet(
                f"color: {color}; border: 1px solid #444; "
                "border-radius: 4px; padding: 6px; font-weight: bold; font-size: 10px;"
            )

            # Chi-squared line
            if result.total_summons > 0:
                self._chi_lbl.setText(
                    f"χ²({result.chi_sq_df}) = {result.chi_sq_statistic}  |  "
                    f"p = {result.chi_sq_p_value}  |  "
                    f"n = {result.total_summons} summons  |  "
                    f"deck size = {result.deck_size}  "
                    + ("" if result.reliable else
                       f"  ⚠ need ≥ {result.MIN_RELIABLE} for reliable results")
                )
            else:
                self._chi_lbl.setText("")

            # Per-unit table
            t = self._summon_unit_table
            t.setRowCount(len(result.unit_stats))
            for i, us in enumerate(result.unit_stats):
                ci_text = f"{us.ci_low:.1%} – {us.ci_high:.1%}"
                vals = [
                    us.display_name,
                    str(us.observed),
                    f"{us.expected:.1f}",
                    f"{us.observed_rate:.1%}",
                    f"{us.expected_rate:.1%}",
                    f"{us.z_score:+.2f}",
                    f"{us.p_value:.4f}",
                    ci_text,
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if us.flagged:
                        item.setForeground(QColor("#d4a017"))
                    t.setItem(i, j, item)

            # Merge table
            m = self._summon_merge_table
            m.setRowCount(len(result.merge_stats))
            for i, ms in enumerate(result.merge_stats):
                for j, v in enumerate([
                    ms.display_name, str(ms.from_rank),
                    str(ms.to_rank), str(ms.count)
                ]):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    m.setItem(i, j, item)

        except Exception as exc:
            self._summon_verdict.setText(f"Error loading stats: {exc}")

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
                # Store full match_id on column 0 for selection lookup
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(row["match_id"]))
                self._hist_table.setItem(i, j, item)

        self._hist_detail_box.setVisible(False)

    def _on_history_selected(self):
        """Populate the summon breakdown panel for whichever match row is selected."""
        selected = self._hist_table.selectedItems()
        if not selected:
            self._hist_detail_box.setVisible(False)
            return

        row_idx = self._hist_table.currentRow()
        id_item = self._hist_table.item(row_idx, 0)
        if id_item is None:
            self._hist_detail_box.setVisible(False)
            return

        match_id = id_item.data(Qt.ItemDataRole.UserRole)
        if not match_id:
            self._hist_detail_box.setVisible(False)
            return

        try:
            from src.database.connection import _DB_PATHS
            import sqlite3 as _sq
            db_path = Path(_DB_PATHS["summon_analysis"])
            if not db_path.exists():
                self._hist_detail_box.setVisible(False)
                return

            conn = _sq.connect(db_path)
            conn.row_factory = _sq.Row

            # Get deck composition for expected rate
            session = conn.execute(
                "SELECT deck_json, total_summons FROM summon_sessions WHERE match_id = ?",
                (match_id,)
            ).fetchone()

            if session is None or not session["deck_json"]:
                self._hist_detail_box.setVisible(False)
                conn.close()
                return

            import json as _json
            deck = _json.loads(session["deck_json"])
            total = session["total_summons"]
            deck_size = len(deck)
            expected_rate = 1.0 / deck_size if deck_size > 0 else 0.0

            counts = {r["unit_summoned"]: r["count"] for r in conn.execute(
                """SELECT unit_summoned, COUNT(*) AS count
                   FROM summon_events WHERE match_id = ?
                   GROUP BY unit_summoned""",
                (match_id,)
            ).fetchall()}
            conn.close()

            t = self._hist_detail_table
            t.setRowCount(len(deck))
            for i, uid in enumerate(sorted(deck)):
                name = self._unit_names.get(uid, uid)
                obs  = counts.get(uid, 0)
                rate = obs / total if total > 0 else 0.0
                dev  = rate - expected_rate
                dev_text = f"{dev:+.1%}"
                if dev > expected_rate * 0.15:
                    dev_color = QColor("#5dba7d")
                elif dev < -expected_rate * 0.15:
                    dev_color = QColor("#ba5d5d")
                else:
                    dev_color = QColor(_TEXT)

                for j, val in enumerate([name, str(obs), f"{rate:.1%}", dev_text]):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if j == 3:
                        item.setForeground(dev_color)
                    t.setItem(i, j, item)

            self._hist_detail_box.setTitle(
                f"Summon Breakdown — {total} summons across {deck_size}-unit deck"
            )
            self._hist_detail_box.setVisible(True)

        except Exception:
            self._hist_detail_box.setVisible(False)

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

    # ------------------------------------------------------------------
    # Placeholder / preview state
    # ------------------------------------------------------------------

    def _show_placeholder_state(self):
        """
        Pre-fill every new UI area with clearly-labelled preview data so the
        layout is visible before any match is recorded.
        Wiped automatically when a real match starts.
        """
        from src.analysis.game_state import UnitCell, BoardState

        # ── Placeholder unit definitions ──────────────────────────────
        _player_defs = [
            ("archer",       "Archer",       3, {1: "R"},          0.95),
            ("crystal_mage", "Crystal Mage", 2, {1: "L"},          0.88),
            ("engineer",     "Engineer",     4, {1: "R", 2: "Fixed"}, 0.92),
            ("frost",        "Frost",        1, {},                 0.75),
            ("treant",       "Treant",       2, {},                 0.60),
        ]
        _opp_defs = [
            ("dark_star", "Dark Star", 3, {1: "L"},   0.90),
            ("bomber",    "Bomber",    2, {},          0.85),
            ("dryad",     "Dryad",     3, {1: "R"},   0.91),
            ("keeper",    "Keeper",    1, {},          0.70),
            ("poisoner",  "Poisoner",  2, {1: "L"},   0.82),
        ]

        # Add placeholder names to the shared unit_names dict so rosters
        # render them (dict is shared by reference with UnitRosterWidget)
        for uid, name, *_ in _player_defs + _opp_defs:
            self._unit_names.setdefault(uid, name)

        # ── Build fake board states ───────────────────────────────────
        def _make_board(defs):
            board = BoardState()
            for i, (uid, _, rank, talents, conf) in enumerate(defs):
                r, c = divmod(i, 3)
                board.set(r, c, UnitCell(
                    unit_id=uid, merge_rank=rank,
                    talent_path=talents, recognition_confidence=conf,
                ))
            return board

        player_board = _make_board(_player_defs)
        opp_board    = _make_board(_opp_defs)

        # ── Fake summon counts (slight variance from expected 20 %) ───
        player_counts = {
            "archer": 28, "crystal_mage": 17, "engineer": 22,
            "frost": 19, "treant": 14,
        }
        opp_counts = {
            "dark_star": 21, "bomber": 30, "dryad": 16,
            "keeper": 20, "poisoner": 13,
        }

        # ── Composition panel ─────────────────────────────────────────
        self._player_roster.update_hero("Paladin  [PREVIEW]")
        self._opp_roster.update_hero("Monk  [PREVIEW]")
        self._player_roster.update_roster(player_board, player_counts)
        self._opp_roster.update_roster(opp_board, opp_counts)

        # ── HUD ───────────────────────────────────────────────────────
        self._wave_lbl.setText("Wave: 12")
        self._php_lbl.setText("Player HP: 3")
        self._ohp_lbl.setText("Opp HP: 2")
        self._mana_lbl.setText("Mana: 180")
        self._conf_lbl.setText("Confidence: —")
        self._frame_lbl.setText("Frames: —")
        self._update_prob_bar(0.63)

        # ── Match History: one fake row + breakdown ───────────────────
        self._hist_table.setRowCount(1)
        for j, val in enumerate(
            ["abcd1234…", "2026-04-17 10:30", "live_capture",
             "Paladin", "15", "win"]
        ):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if j == 5:
                item.setForeground(QColor("#5dba7d"))
            if j == 0:
                item.setData(Qt.ItemDataRole.UserRole, "")  # no real match_id
            self._hist_table.setItem(0, j, item)

        total_s = sum(player_counts.values())
        exp_r   = 1.0 / len(player_counts)
        self._hist_detail_table.setRowCount(len(_player_defs))
        for i, (uid, name, *_) in enumerate(_player_defs):
            obs  = player_counts[uid]
            rate = obs / total_s
            dev  = rate - exp_r
            dtext = f"{dev:+.1%}"
            if dev > exp_r * 0.15:
                dcol = QColor("#5dba7d")
            elif dev < -exp_r * 0.15:
                dcol = QColor("#ba5d5d")
            else:
                dcol = QColor(_TEXT)
            for j, v in enumerate([name, str(obs), f"{rate:.1%}", dtext]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 3:
                    item.setForeground(dcol)
                self._hist_detail_table.setItem(i, j, item)
        self._hist_detail_box.setTitle(
            f"Summon Breakdown — {total_s} summons, 5-unit deck  [PREVIEW DATA]"
        )
        self._hist_detail_box.setVisible(True)

        # ── Summon Stats tab ──────────────────────────────────────────
        self._summon_verdict.setText(
            "Consistent with Fair Randomness  —  p = 0.3812 — no statistically "
            "significant deviation from 20% per unit.  [PREVIEW DATA]"
        )
        self._summon_verdict.setStyleSheet(
            "color: #5dba7d; border: 1px solid #444; border-radius: 4px; "
            "padding: 6px; font-weight: bold; font-size: 10px;"
        )
        self._chi_lbl.setText(
            "χ²(4) = 4.700  |  p = 0.3812  |  n = 100 summons  |  deck size = 5"
        )
        preview_units = [
            ("Archer",       28, 20.0, 0.28, 0.20, +1.43, 0.1527),
            ("Crystal Mage", 17, 20.0, 0.17, 0.20, -1.07, 0.2843),
            ("Engineer",     22, 20.0, 0.22, 0.20, +0.71, 0.4761),
            ("Frost",        19, 20.0, 0.19, 0.20, -0.36, 0.7212),
            ("Treant",       14, 20.0, 0.14, 0.20, -2.14, 0.0328),
        ]
        self._summon_unit_table.setRowCount(len(preview_units))
        for i, (name, obs, exp, obs_r, exp_r2, z, p) in enumerate(preview_units):
            ci_lo = max(0.0, obs_r - 1.96 * (obs_r * (1 - obs_r) / 100) ** 0.5)
            ci_hi = min(1.0, obs_r + 1.96 * (obs_r * (1 - obs_r) / 100) ** 0.5)
            flagged = p < 0.05
            vals = [
                name, str(obs), f"{exp:.1f}",
                f"{obs_r:.0%}", f"{exp_r2:.0%}",
                f"{z:+.2f}", f"{p:.4f}",
                f"{ci_lo:.1%} – {ci_hi:.1%}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if flagged:
                    item.setForeground(QColor("#d4a017"))
                self._summon_unit_table.setItem(i, j, item)

        preview_merges = [
            ("Archer",       2, 3, 12),
            ("Crystal Mage", 1, 2,  8),
            ("Engineer",     3, 4,  6),
            ("Frost",        1, 2,  9),
            ("Treant",       1, 2,  5),
        ]
        self._summon_merge_table.setRowCount(len(preview_merges))
        for i, (name, fr, tr, cnt) in enumerate(preview_merges):
            for j, v in enumerate([name, str(fr), str(tr), str(cnt)]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._summon_merge_table.setItem(i, j, item)

        self._set_status(
            "PREVIEW — placeholder data shown. Start a match to replace with live data."
        )


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