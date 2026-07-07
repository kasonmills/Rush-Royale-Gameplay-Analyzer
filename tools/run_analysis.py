"""
Run full match analysis on a video file and save results to the database.

Usage:
    .venv\\Scripts\\python.exe tools\\run_analysis.py
    .venv\\Scripts\\python.exe tools\\run_analysis.py path\\to\\video.mp4
    .venv\\Scripts\\python.exe tools\\run_analysis.py --outcome win
    .venv\\Scripts\\python.exe tools\\run_analysis.py --outcome loss
    .venv\\Scripts\\python.exe tools\\run_analysis.py --idle-board-sec 60
    .venv\\Scripts\\python.exe tools\\run_analysis.py --idle-board-sec 0

Use --idle-board-sec to extend (or disable with 0) the empty-board timeout when
collecting training data from videos where the reference library is incomplete
and many cells go unrecognised. The default (12s) is correct for real runs.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.analysis.match_runner import MatchRunner, MatchRunnerConfig
from src.analysis.game_state import GameState

_DEFAULT_VIDEO = _ROOT / "data" / "screenshots" / "hud_frames" / "gameplay.mp4"


def on_state(state: GameState):
    occ_p = list(state.player_board.occupied())
    occ_o = list(state.opponent_board.occupied())
    if not (occ_p or occ_o):
        return
    print(f"  t={state.timestamp_sec:5.1f}s | wave={state.wave_number!s:>3}"
          f" | php={state.player_hp!s} ohp={state.opponent_hp!s}"
          f" | conf={state.pipeline_confidence:.2f}"
          f" | win={state.win_probability!s}")
    for r, c, cell in occ_p:
        print(f"    [player   r{r}c{c}] {cell.unit_id}"
              f"  rank={cell.merge_rank}  {cell.appearance_state}"
              f"  conf={cell.recognition_confidence:.2f}")
    for r, c, cell in occ_o:
        print(f"    [opponent r{r}c{c}] {cell.unit_id}"
              f"  rank={cell.merge_rank}  {cell.appearance_state}"
              f"  conf={cell.recognition_confidence:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Analyse a match video and save to DB.")
    ap.add_argument("video", nargs="?", default=str(_DEFAULT_VIDEO))
    ap.add_argument("--outcome", choices=["win", "loss"],
                    help="Record match outcome after analysis")
    ap.add_argument("--sample-every", type=float, default=0.5,
                    help="Seconds between sampled frames (default: 0.5)")
    ap.add_argument("--idle-board-sec", type=float, default=None,
                    help="Seconds of empty board before match is considered over "
                         "(default: 12). Pass 0 to disable entirely.")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        sys.exit(1)

    cfg_kwargs = dict(
        persist=True,
        save_low_confidence=True,
        sample_every_sec=args.sample_every,
    )
    if args.idle_board_sec is not None:
        if args.idle_board_sec == 0:
            cfg_kwargs["idle_detection"] = False
        else:
            cfg_kwargs["idle_empty_board_sec"] = args.idle_board_sec

    cfg = MatchRunnerConfig(**cfg_kwargs)

    print(f"Analysing: {video}")
    print(f"Results will be saved to the database.")
    print(f"Low-confidence crops -> data/to_label/  (for future labelling)")
    if args.idle_board_sec is not None:
        if args.idle_board_sec == 0:
            print(f"Idle detection: DISABLED (training mode)")
        else:
            print(f"Idle board timeout: {args.idle_board_sec}s (default 12s)")
    print()

    runner = MatchRunner.for_video(video, config=cfg)
    result = runner.run(on_state=on_state)

    if args.outcome:
        runner.set_outcome(args.outcome)

    print()
    print("=" * 60)
    print(f"  match_id      : {result.match_id}")
    print(f"  frames proc   : {result.total_frames_processed}")
    print(f"  snapshots saved: {result.total_snapshots_written}")
    print(f"  end_reason    : {result.end_reason}")
    print(f"  final_wave    : {result.final_wave}")
    print(f"  player_deck   : {sorted(result.player_deck) or '(none detected)'}")
    print(f"  opp_deck      : {sorted(result.opponent_deck) or '(none detected)'}")
    print(f"  player_hero   : {result.player_hero_id or '(none)'}")
    print(f"  opp_hero      : {result.opponent_hero_id or '(none)'}")
    if args.outcome:
        print(f"  outcome       : {args.outcome}")
    print("=" * 60)


if __name__ == "__main__":
    main()