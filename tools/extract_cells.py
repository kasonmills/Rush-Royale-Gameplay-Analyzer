"""
Bootstrap cell crop extractor.

Samples frames from a video (or a still image), crops every board cell
(15 player + 15 opponent), and saves non-empty crops to data/to_label/unknown/.

Frames are upscaled to 1080px wide before cropping so each cell comes out
at roughly 170x170px -- usable as a reference image.

After running:
  1. Open data/to_label/unknown/ in Explorer
  2. Sort crops by unit -- move each into data/to_label/<unit_id>/
  3. Rename to the required format and promote to assets/reference/<unit_id>/
     e.g.  base_rank1.png  or  max_level_rank7.png
  4. Re-run the pipeline -- recognition will start firing

Usage:
    .venv\\Scripts\\python.exe tools\\extract_cells.py
    .venv\\Scripts\\python.exe tools\\extract_cells.py path\\to\\video.mp4
    .venv\\Scripts\\python.exe tools\\extract_cells.py path\\to\\screenshot.jpg
    .venv\\Scripts\\python.exe tools\\extract_cells.py --every 5.0
    .venv\\Scripts\\python.exe tools\\extract_cells.py --start 0.10 --end 0.90
    .venv\\Scripts\\python.exe tools\\extract_cells.py --debug
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
import tools.diagnose_video as diag
from src.recognition.board_effect_classifier import BoardEffectClassifier
from src.recognition.hero_classifier import HUDRegions
from src.recognition.ocr_reader import HUDLayout
from src.recognition.template_matcher import TemplateMatcher

_DEFAULT_VIDEO  = _ROOT / "data" / "screenshots" / "hud_frames" / "VID_20260624121536.mp4"
_REF_DIR        = _ROOT / "assets" / "reference"
_OUT_DIR        = _ROOT / "data" / "to_label" / "unknown"
_DEFAULT_EVERY  = 5.0    # seconds between sampled frames
_DEFAULT_START  = 0.05   # skip opening loading screen
_DEFAULT_END    = 0.95   # skip closing results screen
_TARGET_WIDTH   = 1080   # upscale video frames to this width before cropping
_MIN_CROP_SIZE  = 128    # upscale each saved crop to at least this many pixels

# A cell whose grayscale variance is below this is treated as empty background.
_EMPTY_VAR_THRESHOLD = 120.0

# NCC between a cell and its previous capture above which the cell is
# considered unchanged — skip saving a duplicate.
_DEDUP_THRESHOLD = 0.85
_DEDUP_SIZE      = (64, 64)   # downsample for fast cross-frame comparison

# NCC confidence at or above which a cell is confidently identified as a
# catalogued unit — skip saving.
_KNOWN_CONF_THRESHOLD = 0.50

# Soft NCC floor for "probably a catalogued unit in an uncatalogued state."
# When masked NCC is in [_CATALOGUED_UNIT_THRESHOLD, _KNOWN_CONF_THRESHOLD),
# the crop is still skipped — the unit is already in the library so we don't
# need more samples of it.  Only crops that score below this against the entire
# library are saved (genuinely uncatalogued units).
_CATALOGUED_UNIT_THRESHOLD = 0.35

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _cell_unchanged(crop: np.ndarray, last: np.ndarray) -> bool:
    """True when crop is visually near-identical to the previous capture of
    the same cell — no new information worth saving."""
    a = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), _DEDUP_SIZE)
    b = cv2.resize(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), _DEDUP_SIZE)
    score = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score >= _DEDUP_THRESHOLD


def _is_empty(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray)) < _EMPTY_VAR_THRESHOLD


def _upscale(crop: np.ndarray, min_size: int) -> np.ndarray:
    """Upscale crop so its shortest side is at least min_size pixels."""
    h, w = crop.shape[:2]
    if min(h, w) >= min_size:
        return crop
    scale = min_size / min(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _save_crop(crop: np.ndarray, out_dir: Path, stem: str,
               side: str, row: int, col: int,
               ts: float, min_size: int) -> None:
    crop = _upscale(crop, min_size)
    name = f"{stem}_{side}_r{row}c{col}_t{ts:.1f}.png"
    cv2.imwrite(str(out_dir / name), crop)


def _process_frame(frame: np.ndarray, cal: GridCalibrator,
                   out_dir: Path, stem: str, ts: float,
                   include_empty: bool, min_size: int,
                   matcher: "TemplateMatcher | None" = None,
                   effect_clf: "BoardEffectClassifier | None" = None,
                   cell_states: "dict | None" = None,
                   ) -> tuple[int, int, int, int, int]:
    """Crop all 30 cells from one frame.

    Returns (saved, skipped_dark, skipped_dup, skipped_tile, skipped_known).
    """
    saved = skipped_dark = skipped_dup = skipped_tile = skipped_known = 0
    for side in ("player", "opponent"):
        for row in range(cal._data.rows):
            for col in range(cal._data.cols):
                crop = cal.crop_cell(frame, side, row, col)
                key  = (side, row, col)

                # 1. Cross-frame dedup — skip if cell looks identical to last
                #    time we saw something different here.
                if cell_states is not None and key in cell_states:
                    if _cell_unchanged(crop, cell_states[key]):
                        skipped_dup += 1
                        continue
                if cell_states is not None:
                    cell_states[key] = crop.copy()

                # 2. Near-black / featureless cell
                if not include_empty and _is_empty(crop):
                    skipped_dark += 1
                    continue

                # 3. Known empty-tile visual (wind tile, relic tile, sacrifice
                #    tile, kobold mine, etc.)
                if effect_clf is not None and effect_clf.is_empty_tile(crop, threshold=0.75):
                    skipped_tile += 1
                    continue

                # 4. Skip crops from catalogued units.
                #    - Strong match (>= _KNOWN_CONF_THRESHOLD): confirmed unit.
                #    - Weak match (>= _CATALOGUED_UNIT_THRESHOLD): probably the
                #      same unit in an uncatalogued state/rank — skip it since
                #      the unit is already in the library.
                #    Only crops that score below _CATALOGUED_UNIT_THRESHOLD
                #    against the entire library are saved (new units).
                if matcher is not None:
                    result = matcher.match_cell(crop)
                    if result.unit_id is not None or \
                            result.confidence >= _CATALOGUED_UNIT_THRESHOLD:
                        skipped_known += 1
                        continue

                _save_crop(crop, out_dir, stem, side, row, col, ts, min_size)
                saved += 1
    return saved, skipped_dark, skipped_dup, skipped_tile, skipped_known


def _save_debug_frame(frame: np.ndarray, cal: GridCalibrator,
                      out_dir: Path, stem: str, ts: float) -> None:
    """Save one annotated frame showing exactly where each cell was cropped."""
    hud    = HUDLayout()
    heroes = HUDRegions()
    ann    = diag.annotate(frame, cal, hud, heroes)
    diag._legend(ann)
    name   = f"_debug_{stem}_t{ts:.1f}.png"
    cv2.imwrite(str(out_dir / name), ann)
    print(f"  [debug] annotated frame saved: {name}")


def _run_on_image(path: Path, args, out_dir: Path,
                  matcher, effect_clf) -> tuple[int, int, int, int, int]:
    """Extract cells from a single still image."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"ERROR: cannot read image: {path}")
        return 0, 0, 0, 0, 0

    fh, fw = img.shape[:2]
    if fw < _TARGET_WIDTH:
        scale = _TARGET_WIDTH / fw
        img   = cv2.resize(img, (int(fw * scale), int(fh * scale)),
                           interpolation=cv2.INTER_CUBIC)
        fh, fw = img.shape[:2]

    cal  = GridCalibrator.from_defaults(fw, fh)
    stem = path.stem.replace(" ", "_")

    print(f"Image  : {path.name}  ({fw}x{fh})")

    if args.debug:
        _save_debug_frame(img, cal, out_dir, stem, 0.0)

    result = _process_frame(img, cal, out_dir, stem, 0.0,
                            args.include_empty, args.min_size,
                            matcher, effect_clf, cell_states=None)
    saved, s_dark, s_dup, s_tile, s_known = result
    print(f"  -> {saved} saved  |  {s_dark} dark  {s_dup} dup  {s_tile} tile  {s_known} known")
    return result


def _run_on_video(path: Path, args, out_dir: Path,
                  matcher, effect_clf) -> tuple[int, int, int, int, int]:
    """Extract cells from sampled frames of a video."""
    totals = [0, 0, 0, 0, 0]   # saved, dark, dup, tile, known
    debug_saved = False
    cell_states: dict = {}   # (side, row, col) -> last crop that looked different

    with VideoCapture(path, target_width=_TARGET_WIDTH) as cap:
        cap.detect_game_region()

        first = cap.frame_at(0.0)
        if first is None:
            print("ERROR: could not read first frame.")
            return 0, 0, 0, 0, 0

        fh, fw = first.shape[:2]
        cal    = GridCalibrator.from_defaults(fw, fh)
        dur    = cap.duration_sec
        stem   = path.stem.replace(" ", "_")

        start_sec = args.start * dur
        end_sec   = args.end   * dur

        print(f"Frame size : {fw}x{fh}  (upscaled from native)")
        print(f"Duration   : {dur:.1f}s")
        print(f"Sampling   : t={start_sec:.1f}s to t={end_sec:.1f}s  "
              f"every {args.every}s")
        print()

        ts = start_sec
        while ts <= end_sec:
            frame = cap.frame_at(ts)
            if frame is None:
                ts += args.every
                continue

            if args.debug and not debug_saved:
                _save_debug_frame(frame, cal, out_dir, stem, ts)
                debug_saved = True

            saved, s_dark, s_dup, s_tile, s_known = _process_frame(
                frame, cal, out_dir, stem, ts,
                args.include_empty, args.min_size,
                matcher, effect_clf, cell_states,
            )
            for i, v in enumerate((saved, s_dark, s_dup, s_tile, s_known)):
                totals[i] += v

            print(f"  t={ts:6.1f}s  ->  {saved:2d} saved"
                  f"  ({s_dark} dark  {s_dup} dup  {s_tile} tile  {s_known} known  skipped)")
            ts += args.every

    return tuple(totals)


def main():
    ap = argparse.ArgumentParser(
        description="Extract board cell crops from a video or image for labelling."
    )
    ap.add_argument("source", nargs="?", default=str(_DEFAULT_VIDEO),
                    help="Video file or still image (default: gameplay.mp4)")
    ap.add_argument("--every", type=float, default=_DEFAULT_EVERY,
                    metavar="SEC",
                    help=f"Seconds between sampled frames (default: {_DEFAULT_EVERY})")
    ap.add_argument("--start", type=float, default=_DEFAULT_START,
                    metavar="F",
                    help=f"Fraction of video to start from (default: {_DEFAULT_START})")
    ap.add_argument("--end", type=float, default=_DEFAULT_END,
                    metavar="F",
                    help=f"Fraction of video to stop at (default: {_DEFAULT_END})")
    ap.add_argument("--min-size", type=int, default=_MIN_CROP_SIZE,
                    metavar="PX",
                    help=f"Minimum crop dimension in pixels (default: {_MIN_CROP_SIZE})")
    ap.add_argument("--out-dir", default=str(_OUT_DIR),
                    help="Output directory (default: data/to_label/unknown)")
    ap.add_argument("--include-empty", action="store_true",
                    help="Save empty-looking cells too (off by default)")
    ap.add_argument("--debug", action="store_true",
                    help="Save one annotated frame showing the calibration grid overlay")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: source not found: {source}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load classifiers — used to skip crops that are already identified or are
    # known empty/tile visuals (wind tiles, relic tiles, kobold mines, etc.)
    matcher = TemplateMatcher(cell_threshold=_KNOWN_CONF_THRESHOLD,
                              use_rotation=True, use_shape_mask=True)
    if _REF_DIR.is_dir():
        matcher.load_library(_REF_DIR)
    else:
        print(f"WARNING: reference dir not found at {_REF_DIR} — skipping known-unit filter")
        matcher = None

    effect_clf = BoardEffectClassifier()
    if _REF_DIR.is_dir():
        effect_clf.load(_REF_DIR)
    else:
        effect_clf = None

    print(f"Source : {source}")
    print(f"Output : {out_dir}")
    print(f"Min crop size  : {args.min_size}px")
    print(f"Known threshold: {_KNOWN_CONF_THRESHOLD} NCC  (crops above this are skipped)")
    print()

    if source.suffix.lower() in _IMAGE_EXTS:
        saved, s_dark, s_dup, s_tile, s_known = _run_on_image(source, args, out_dir,
                                                               matcher, effect_clf)
    else:
        saved, s_dark, s_dup, s_tile, s_known = _run_on_video(source, args, out_dir,
                                                               matcher, effect_clf)

    total_skipped = s_dark + s_dup + s_tile + s_known
    print()
    print("=" * 60)
    print(f"  Crops saved        : {saved}  ->  {out_dir}")
    print(f"  Skipped (dup)      : {s_dup}   identical to previous capture of same cell")
    print(f"  Skipped (dark)     : {s_dark}   near-black / featureless cells")
    print(f"  Skipped (tile)     : {s_tile}   matched a known empty-tile reference")
    print(f"  Skipped (known)    : {s_known}   already identified at >={_KNOWN_CONF_THRESHOLD} NCC")
    print(f"  Total skipped      : {total_skipped}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Open the output folder in Explorer")
    print("  2. Sort crops by unit -- move into data/to_label/<unit_id>/")
    print("  3. Rename to  base_rank1.png  or  max_level_rank7.png  etc.")
    print("  4. Promote best examples to  assets/reference/<unit_id>/")
    print("  5. Re-run the pipeline -- recognition will start firing")


if __name__ == "__main__":
    main()