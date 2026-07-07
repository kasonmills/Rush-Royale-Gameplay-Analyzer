"""
Dry-run MatchRunner against gameplay.mp4.
No database writes (persist=False). Prints per-frame state to stdout.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.analysis.match_runner import MatchRunner, MatchRunnerConfig
from src.analysis.game_state import GameState

VIDEO = _ROOT / "data" / "screenshots" / "hud_frames" / "gameplay.mp4"

cfg = MatchRunnerConfig(
    persist=False,
    save_low_confidence=False,   # don't write crops during dry run
    idle_detection=False,        # process the full video regardless
    sample_every_sec=2.0,        # one frame every 2s — fast enough to validate
)

frame_count = [0]

def on_state(state: GameState):
    frame_count[0] += 1
    n = frame_count[0]
    occ_p = list(state.player_board.occupied())
    occ_o = list(state.opponent_board.occupied())
    any_recognized = occ_p or occ_o
    if n % 5 == 1 or n <= 3 or any_recognized:
        print(
            f"  frame {n:3d} | t={state.timestamp_sec:5.1f}s"
            f" | wave={state.wave_number!s:>3}"
            f" | php={state.player_hp!s:>2} ohp={state.opponent_hp!s:>2}"
            f" | board: {len(occ_p)}p / {len(occ_o)}o cells occupied"
            f" | conf={state.pipeline_confidence:.2f}"
            f" | win_prob={state.win_probability!s}"
        )
        for r, c, cell in occ_p:
            print(f"    [player  r{r}c{c}] {cell.unit_id}  rank={cell.merge_rank}"
                  f"  appearance={cell.appearance_state}  conf={cell.recognition_confidence:.2f}")
        for r, c, cell in occ_o:
            print(f"    [opponent r{r}c{c}] {cell.unit_id}  rank={cell.merge_rank}"
                  f"  appearance={cell.appearance_state}  conf={cell.recognition_confidence:.2f}")

print(f"[dry_run] Video: {VIDEO}")
print(f"[dry_run] Sampling every {cfg.sample_every_sec}s, persist=False\n")

runner = MatchRunner.for_video(VIDEO, config=cfg)
result = runner.run(on_state=on_state)

print()
print("=" * 60)
print(f"  match_id     : {result.match_id}")
print(f"  frames proc  : {result.total_frames_processed}")
print(f"  snapshots    : {result.total_snapshots_written}  (persist=False → 0 expected)")
print(f"  end_reason   : {result.end_reason}")
print(f"  duration     : {result.duration_sec:.1f}s wall time")
print(f"  final_wave   : {result.final_wave}")
print(f"  player_deck  : {sorted(result.player_deck) or '(none detected)'}")
print(f"  opp_deck     : {sorted(result.opponent_deck) or '(none detected)'}")
print(f"  player_hero  : {result.player_hero_id or '(none)'}")
print(f"  opp_hero     : {result.opponent_hero_id or '(none)'}")
print("=" * 60)
print("[dry_run] Done — no database changes were made.")