"""
Organize raw extracted sprites into the reference library structure.

After running tools/extract_assets.py on the Rush Royale APK, all sprites
land in assets/raw/. This tool maps them to the structure expected by the
recognition pipeline:

  assets/reference/<unit_id>/<appearance_state>_rank<N>[_<variant>].png
  assets/reference/talent_icons/<tier>_<branch>.png
  assets/reference/hero_portraits/<hero_id>.png

Then updates visual_reference.db so the pipeline knows which entries have
been captured.

Recommended two-phase workflow:
  Phase 1 — Review proposed auto-mappings (no files changed):
      python tools/organize_assets.py

  Phase 2a — Happy with the auto-mappings? Apply directly:
      python tools/organize_assets.py --apply

  Phase 2b — Need to correct some mappings first?
      python tools/organize_assets.py --generate-mapping mappings.json
      # Edit mappings.json in any text editor
      python tools/organize_assets.py --apply-mapping mappings.json

The JSON mapping file is the primary correction mechanism. Every entry the
auto-matcher gets wrong can be fixed there before anything is copied.

Options:
    --raw-dir DIR             Source directory (default: assets/raw)
    --ref-dir DIR             Destination root (default: assets/reference)
    --apply                   Copy files and update visual_reference.db
    --move                    Move instead of copy (only with --apply)
    --generate-mapping FILE   Write proposed mappings to a JSON file for editing
    --apply-mapping FILE      Apply a (possibly hand-edited) JSON mapping file
    --unit UNIT_ID            Only process assets whose matched unit is UNIT_ID
    --filter TEXT             Only consider assets whose filename contains TEXT
    --min-confidence FLOAT    Fuzzy-match threshold, default 0.55
"""

import argparse
import csv
import difflib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR   = REPO_ROOT / "assets" / "raw"
REF_DIR   = REPO_ROOT / "assets" / "reference"
SUMMARY   = RAW_DIR / "_extraction_summary.csv"

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Asset category constants
# ---------------------------------------------------------------------------

CATEGORY_UNIT    = "unit"
CATEGORY_TALENT  = "talent"
CATEGORY_HERO    = "hero"
CATEGORY_UNKNOWN = "unknown"

# Known variant tags (Twins: moon/sun; Enchanted Sword: blue/red)
_VARIANT_KEYWORDS = ["moon", "sun", "blue", "red"]

# Appearance state detection — checked in priority order (most specific first)
_APPEARANCE_MAP = [
    ("reincarnation_3", ["reinc3", "reborn3", "evo3", "reincarnation3"]),
    ("reincarnation_2", ["reinc2", "reborn2", "evo2", "reincarnation2"]),
    ("reincarnation_1", ["reinc1", "reborn1", "evo1", "reincarnation1",
                         "reincarnation", "reborn", "rebirth"]),
    ("max_level",       ["max_level", "maxlevel", "max_lvl", "maxlvl", "maxed"]),
]

# Default rank when a max_level asset has no explicit rank
_MAX_RANK_DEFAULT = 7

# Talent tier/branch detection
_TALENT_BRANCHES = {
    "l": "L", "left": "L",
    "r": "R", "right": "R",
    "fixed": "Fixed", "fix": "Fixed",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AssetMatch:
    raw_file:         str
    category:         str              # 'unit', 'talent', 'hero', 'unknown'
    unit_id:          Optional[str]
    appearance_state: str              # 'base', 'max_level', 'reincarnation_N'
    merge_rank:       Optional[int]    # 1–7 for units; None for others
    variant_tag:      Optional[str]    # 'moon', 'sun', 'blue', 'red', etc.
    talent_tier:      Optional[int]    # 1–4 (talent icons only)
    talent_branch:    Optional[str]    # 'L', 'R', 'Fixed' (talent icons only)
    hero_id:          Optional[str]    # hero_id (hero portraits only)
    confidence:       float            # fuzzy match score 0.0–1.0
    dest_rel:         Optional[str]    # relative destination path from ref_dir


# ---------------------------------------------------------------------------
# Helpers — name parsing
# ---------------------------------------------------------------------------

def _tokens(name: str) -> list[str]:
    """Split a stem into lowercase tokens on _, -, space."""
    return re.split(r"[_\-\s]+", name.lower())


def detect_appearance_state(tokens: list[str]) -> str:
    joined = "_".join(tokens)
    for state, keywords in _APPEARANCE_MAP:
        for kw in keywords:
            if kw in joined:
                return state
    return "base"


def detect_rank(tokens: list[str]) -> Optional[int]:
    """
    Extract merge rank from token list.  Returns the first digit 1–7 found
    after an explicit rank-marker token ('rank', 'r', 'lvl', 'level'),
    then falls back to any standalone digit 1–7 at the end of the token list.
    """
    # Pass 1: explicit marker followed by a digit
    for i, tok in enumerate(tokens):
        if tok in ("rank", "r", "lvl", "level") and i + 1 < len(tokens):
            try:
                n = int(tokens[i + 1])
                if 1 <= n <= 7:
                    return n
            except ValueError:
                pass
        # marker fused with digit, e.g. "rank1", "r3", "lvl7"
        m = re.match(r"(?:rank|lvl?|r)(\d)$", tok)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 7:
                return n

    # Pass 2: trailing standalone digit
    for tok in reversed(tokens):
        if re.fullmatch(r"\d", tok):
            n = int(tok)
            if 1 <= n <= 7:
                return n

    return None


def detect_variant(tokens: list[str]) -> Optional[str]:
    for tok in tokens:
        if tok in _VARIANT_KEYWORDS:
            return tok
    return None


def _clean_for_matching(tokens: list[str]) -> str:
    """
    Remove rank, appearance, variant, and common generic prefixes so the
    remaining text is as close as possible to the unit name.
    """
    skip = {
        # rank markers
        "rank", "r", "lvl", "level",
        # appearance markers
        "max", "maxlevel", "maxlvl", "maxed", "base",
        "reincarnation", "reinc", "reborn", "rebirth", "evo",
        # generic game prefixes
        "card", "unit", "character", "sprite", "icon", "img", "image",
        "hero", "portrait", "talent", "badge",
        # branches
        "left", "right", "fixed",
        # common noise
        "new", "old", "v2", "v3", "alt",
    }
    kept = []
    for tok in tokens:
        # drop pure digit tokens
        if re.fullmatch(r"\d+", tok):
            continue
        # drop known skip words
        if tok in skip:
            continue
        # drop variant keywords
        if tok in _VARIANT_KEYWORDS:
            continue
        kept.append(tok)
    return " ".join(kept)


def _best_unit_match(cleaned: str,
                     units: list[tuple[str, str]],
                     min_confidence: float) -> tuple[Optional[str], float]:
    """
    Fuzzy-match `cleaned` against (unit_id, display_name) pairs.
    Returns (unit_id, score) or (None, 0.0).
    """
    if not cleaned or not units:
        return None, 0.0

    best_id, best_score = None, 0.0
    for uid, display in units:
        # compare against both the display name and the unit_id
        for candidate in (display.lower(), uid.lower().replace("_", " ")):
            score = difflib.SequenceMatcher(None, cleaned, candidate).ratio()
            if score > best_score:
                best_score = score
                best_id = uid

    return (best_id, best_score) if best_score >= min_confidence else (None, best_score)


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

def _detect_category(tokens: list[str]) -> str:
    joined = "_".join(tokens)
    if "talent" in joined or "badge" in joined:
        return CATEGORY_TALENT
    if "hero" in joined or "portrait" in joined:
        return CATEGORY_HERO
    return CATEGORY_UNIT   # default; caller may override if match fails


def _detect_talent(tokens: list[str]) -> tuple[Optional[int], Optional[str]]:
    """Extract (tier, branch) from a talent asset's token list."""
    tier: Optional[int] = None
    branch: Optional[str] = None

    for i, tok in enumerate(tokens):
        if tok in ("tier", "t") and i + 1 < len(tokens):
            try:
                n = int(tokens[i + 1])
                if 1 <= n <= 4:
                    tier = n
            except ValueError:
                pass
        m = re.match(r"t(\d)$", tok)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 4:
                tier = n
        if tok in _TALENT_BRANCHES:
            branch = _TALENT_BRANCHES[tok]

    # Fallback: single digit 1-4 anywhere
    if tier is None:
        for tok in tokens:
            if re.fullmatch(r"[1-4]", tok):
                tier = int(tok)
                break

    return tier, branch


# ---------------------------------------------------------------------------
# Build destination path
# ---------------------------------------------------------------------------

def _dest_rel(match: AssetMatch) -> Optional[str]:
    if match.category == CATEGORY_UNIT and match.unit_id and match.merge_rank:
        stem = f"{match.appearance_state}_rank{match.merge_rank}"
        if match.variant_tag:
            stem += f"_{match.variant_tag}"
        return f"{match.unit_id}/{stem}.png"

    if match.category == CATEGORY_TALENT and match.talent_tier and match.talent_branch:
        return f"talent_icons/{match.talent_tier}_{match.talent_branch}.png"

    if match.category == CATEGORY_HERO and match.hero_id:
        return f"hero_portraits/{match.hero_id}.png"

    return None


# ---------------------------------------------------------------------------
# Main matching logic
# ---------------------------------------------------------------------------

def build_match(entry: dict,
                units: list[tuple[str, str]],
                heroes: list[tuple[str, str]],
                min_confidence: float) -> AssetMatch:
    filename = entry["file"]
    stem     = Path(filename).stem
    toks     = _tokens(stem)

    category = _detect_category(toks)

    # --- Talent icon ---
    if category == CATEGORY_TALENT:
        tier, branch = _detect_talent(toks)
        m = AssetMatch(
            raw_file=filename, category=CATEGORY_TALENT,
            unit_id=None, appearance_state="", merge_rank=None,
            variant_tag=None, talent_tier=tier, talent_branch=branch,
            hero_id=None, confidence=1.0 if (tier and branch) else 0.0,
            dest_rel=None,
        )
        m.dest_rel = _dest_rel(m)
        return m

    # --- Hero portrait ---
    if category == CATEGORY_HERO:
        cleaned = _clean_for_matching(toks)
        hero_id, conf = _best_unit_match(cleaned, heroes, min_confidence)
        m = AssetMatch(
            raw_file=filename, category=CATEGORY_HERO,
            unit_id=None, appearance_state="", merge_rank=None,
            variant_tag=None, talent_tier=None, talent_branch=None,
            hero_id=hero_id, confidence=conf,
            dest_rel=None,
        )
        m.dest_rel = _dest_rel(m)
        return m

    # --- Unit sprite (default) ---
    appearance = detect_appearance_state(toks)
    rank       = detect_rank(toks)
    variant    = detect_variant(toks)
    cleaned    = _clean_for_matching(toks)
    unit_id, conf = _best_unit_match(cleaned, units, min_confidence)

    # If a max_level asset has no rank, default to 7
    if appearance == "max_level" and rank is None:
        rank = _MAX_RANK_DEFAULT

    m = AssetMatch(
        raw_file=filename, category=CATEGORY_UNIT,
        unit_id=unit_id, appearance_state=appearance,
        merge_rank=rank, variant_tag=variant,
        talent_tier=None, talent_branch=None, hero_id=None,
        confidence=conf, dest_rel=None,
    )
    m.dest_rel = _dest_rel(m)
    return m


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_units() -> list[tuple[str, str]]:
    """Load (unit_id, display_name) from unit_meta.db, or return empty list."""
    try:
        from src.database.connection import unit_meta_db
        with unit_meta_db() as conn:
            rows = conn.execute(
                "SELECT unit_id, display_name FROM units ORDER BY unit_id"
            ).fetchall()
            return [(r["unit_id"], r["display_name"]) for r in rows]
    except Exception:
        return []


def _load_heroes() -> list[tuple[str, str]]:
    """Load (hero_id, display_name) from unit_meta.db, or return empty list."""
    try:
        from src.database.connection import unit_meta_db
        with unit_meta_db() as conn:
            rows = conn.execute(
                "SELECT hero_id, display_name FROM heroes ORDER BY hero_id"
            ).fetchall()
            return [(r["hero_id"], r["display_name"]) for r in rows]
    except Exception:
        return []


def _update_visual_ref_db(matches: list[AssetMatch]):
    """Upsert captured entries into visual_reference.db."""
    try:
        from src.database.connection import visual_ref_db
        from src.database.visual_ref_repo import VisualRefRepo, HeroPortraitRefRepo
    except ImportError:
        print("  [WARN] Could not import DB modules — visual_reference.db not updated.")
        return

    unit_rows, hero_rows = [], []
    for m in matches:
        if not m.dest_rel:
            continue
        if m.category == CATEGORY_UNIT and m.unit_id and m.merge_rank:
            unit_rows.append({
                "unit_id":          m.unit_id,
                "appearance_state": m.appearance_state,
                "merge_rank":       m.merge_rank,
                "variant_tag":      m.variant_tag,
                "file_path":        f"assets/reference/{m.dest_rel}",
                "captured":         1,
            })
        elif m.category == CATEGORY_HERO and m.hero_id:
            hero_rows.append({
                "hero_id":   m.hero_id,
                "file_path": f"assets/reference/{m.dest_rel}",
                "captured":  1,
            })

    try:
        with visual_ref_db() as conn:
            if unit_rows:
                VisualRefRepo.upsert_many(conn, unit_rows)
            if hero_rows:
                HeroPortraitRefRepo.upsert_many(conn, hero_rows)
        print(f"  Updated visual_reference.db: "
              f"{len(unit_rows)} unit entries, {len(hero_rows)} hero entries.")
    except Exception as exc:
        print(f"  [WARN] DB update failed: {exc}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_dry_run(matches: list[AssetMatch]):
    matched   = [m for m in matches if m.dest_rel]
    unmatched = [m for m in matches if not m.dest_rel]

    # Group matched by unit / category
    by_unit: dict[str, list[AssetMatch]] = {}
    for m in matched:
        key = (m.unit_id or m.hero_id or
               f"talent_icons/T{m.talent_tier}") or "?"
        by_unit.setdefault(key, []).append(m)

    print(f"\n{'='*70}")
    print(f" Proposed mappings — {len(matched)} matched, {len(unmatched)} unmatched")
    print(f"{'='*70}\n")

    for key in sorted(by_unit):
        group = sorted(by_unit[key], key=lambda m: (m.appearance_state, m.merge_rank or 0))
        print(f"  {key}/")
        for m in group:
            dest = m.dest_rel.split("/", 1)[1] if "/" in (m.dest_rel or "") else m.dest_rel
            conf = f"{m.confidence:.0%}"
            print(f"    {dest:<40}  ←  {m.raw_file:<35} [{conf}]")
        print()

    if unmatched:
        print(f"  UNMATCHED ({len(unmatched)} assets — will be skipped):")
        for m in unmatched:
            print(f"    {m.raw_file}  (best score: {m.confidence:.0%})")
        print()

    print("To apply:              python tools/organize_assets.py --apply")
    print("To export for editing: python tools/organize_assets.py "
          "--generate-mapping mappings.json")


def _write_mapping_json(matches: list[AssetMatch], path: Path):
    data = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total":    len(matches),
            "matched":  sum(1 for m in matches if m.dest_rel),
            "unmatched":sum(1 for m in matches if not m.dest_rel),
            "instructions": (
                "Edit unit_id, appearance_state, merge_rank, variant_tag, "
                "talent_tier, talent_branch, or hero_id to fix wrong matches. "
                "Delete an entry to skip that asset. "
                "dest_rel is auto-computed on apply — you may leave it null."
            ),
        },
        "mappings": [],
        "unmatched": [],
    }

    for m in matches:
        entry = {
            "raw_file":         m.raw_file,
            "category":         m.category,
            "unit_id":          m.unit_id,
            "appearance_state": m.appearance_state or "base",
            "merge_rank":       m.merge_rank,
            "variant_tag":      m.variant_tag,
            "talent_tier":      m.talent_tier,
            "talent_branch":    m.talent_branch,
            "hero_id":          m.hero_id,
            "confidence":       round(m.confidence, 3),
            "dest_rel":         m.dest_rel,
        }
        if m.dest_rel:
            data["mappings"].append(entry)
        else:
            data["unmatched"].append(entry)

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Mapping file written to {path}")
    print(f"Edit it, then run: python tools/organize_assets.py --apply-mapping {path}")


# ---------------------------------------------------------------------------
# Apply helpers
# ---------------------------------------------------------------------------

def _apply_match_list(matches: list[AssetMatch],
                      raw_dir: Path, ref_dir: Path,
                      move: bool, update_db: bool):
    applicable = [m for m in matches if m.dest_rel]
    if not applicable:
        print("Nothing to apply.")
        return

    copied, skipped = 0, 0
    for m in applicable:
        src  = raw_dir / m.raw_file
        dest = ref_dir / m.dest_rel

        if not src.exists():
            print(f"  [SKIP] Source not found: {src}")
            skipped += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), dest)
        else:
            shutil.copy2(src, dest)
        copied += 1

    verb = "Moved" if move else "Copied"
    print(f"{verb} {copied} files to {ref_dir}  ({skipped} skipped).")

    if update_db:
        _update_visual_ref_db(applicable)


def _apply_from_json(mapping_path: Path,
                     raw_dir: Path, ref_dir: Path,
                     move: bool, update_db: bool):
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    entries = data.get("mappings", []) + [
        e for e in data.get("unmatched", []) if e.get("dest_rel")
    ]
    if not entries:
        print("No applicable entries found in mapping file.")
        return

    matches = []
    for e in entries:
        m = AssetMatch(
            raw_file=e["raw_file"],
            category=e.get("category", CATEGORY_UNIT),
            unit_id=e.get("unit_id"),
            appearance_state=e.get("appearance_state") or "base",
            merge_rank=e.get("merge_rank"),
            variant_tag=e.get("variant_tag"),
            talent_tier=e.get("talent_tier"),
            talent_branch=e.get("talent_branch"),
            hero_id=e.get("hero_id"),
            confidence=e.get("confidence", 0.0),
            dest_rel=e.get("dest_rel"),
        )
        # Recompute dest_rel if missing or if the user edited underlying fields
        if not m.dest_rel:
            m.dest_rel = _dest_rel(m)
        matches.append(m)

    _apply_match_list(matches, raw_dir, ref_dir, move, update_db)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Organize raw extracted sprites into the reference library."
    )
    parser.add_argument("--raw-dir",   default=str(RAW_DIR),
                        help="Source directory (default: assets/raw)")
    parser.add_argument("--ref-dir",   default=str(REF_DIR),
                        help="Destination root (default: assets/reference)")
    parser.add_argument("--apply",     action="store_true",
                        help="Copy files and update visual_reference.db")
    parser.add_argument("--move",      action="store_true",
                        help="Move instead of copy (only with --apply)")
    parser.add_argument("--generate-mapping", metavar="FILE",
                        help="Export proposed mappings to a JSON file for editing")
    parser.add_argument("--apply-mapping", metavar="FILE",
                        help="Apply a (possibly hand-edited) JSON mapping file")
    parser.add_argument("--unit",      metavar="UNIT_ID",
                        help="Only process assets for this unit_id")
    parser.add_argument("--filter",    metavar="TEXT",
                        help="Only consider assets whose filename contains TEXT")
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        metavar="FLOAT",
                        help="Fuzzy match threshold (default: 0.55)")
    parser.add_argument("--no-db",     action="store_true",
                        help="Skip visual_reference.db update even when applying")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    ref_dir = Path(args.ref_dir)

    # --apply-mapping is a self-contained path — no need to build matches first
    if args.apply_mapping:
        mapping_path = Path(args.apply_mapping)
        if not mapping_path.exists():
            sys.exit(f"Mapping file not found: {mapping_path}")
        print(f"Applying mapping file: {mapping_path}")
        _apply_from_json(mapping_path, raw_dir, ref_dir,
                         move=args.move, update_db=not args.no_db)
        return

    # Load extraction summary
    if not SUMMARY.exists():
        sys.exit(
            f"Extraction summary not found at {SUMMARY}.\n"
            "Run tools/extract_assets.py first to populate assets/raw/."
        )

    with open(SUMMARY, newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    if not entries:
        sys.exit("Extraction summary is empty — nothing to organise.")

    # Apply name filter early to reduce work
    if args.filter:
        entries = [e for e in entries if args.filter.lower() in e["file"].lower()]

    # Load known units and heroes from DB (gracefully empty if DB not ready)
    units  = _load_units()
    heroes = _load_heroes()

    if not units:
        print(
            "[WARN] No units found in unit_meta.db. "
            "Run tools/sync_from_sheet.py first for best matching results.\n"
            "Continuing with name-only heuristics.\n"
        )

    print(f"Processing {len(entries)} extracted assets...")

    matches = [
        build_match(e, units, heroes, args.min_confidence)
        for e in entries
    ]

    # Post-filter by --unit
    if args.unit:
        matches = [
            m for m in matches
            if m.unit_id == args.unit or m.hero_id == args.unit
        ]

    if args.generate_mapping:
        _write_mapping_json(matches, Path(args.generate_mapping))
        return

    if args.apply:
        _apply_match_list(matches, raw_dir, ref_dir,
                          move=args.move, update_db=not args.no_db)
        return

    # Default: dry-run
    _print_dry_run(matches)


if __name__ == "__main__":
    main()
