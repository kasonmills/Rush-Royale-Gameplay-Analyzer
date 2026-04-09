"""
Extract sprite and texture assets from a Rush Royale APK or Unity asset bundle.

Supports:
  - .apk files (treated as a zip; scans all internal asset bundles)
  - .obb files (same as apk)
  - A directory of asset bundle files (scans recursively)
  - A single Unity asset bundle file

Output:
  All Sprite and Texture2D objects are saved as PNGs under:
    assets/raw/<sanitized_asset_name>.png

  A summary CSV is written to:
    assets/raw/_extraction_summary.csv

Usage:
    python tools/extract_assets.py <path_to_apk_or_dir>

    # Optional: filter by name substring
    python tools/extract_assets.py rushroyal.apk --filter card

Requirements:
    pip install UnityPy Pillow
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path

try:
    import UnityPy
except ImportError:
    sys.exit("UnityPy not installed. Run: pip install UnityPy")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed. Run: pip install Pillow")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "assets" / "raw"


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def extract_from_bundle(env, output_dir: Path, name_filter: str | None) -> list[dict]:
    records = []
    for obj in env.objects:
        if obj.type.name not in ("Sprite", "Texture2D"):
            continue

        data = obj.read()
        asset_name = getattr(data, "name", None) or f"unnamed_{obj.path_id}"

        if name_filter and name_filter.lower() not in asset_name.lower():
            continue

        try:
            if obj.type.name == "Sprite":
                image: Image.Image = data.image
            else:
                image: Image.Image = data.image

            safe_name = sanitize(asset_name)
            out_path = output_dir / f"{safe_name}.png"

            # Avoid overwriting if duplicate names exist across bundles
            counter = 1
            while out_path.exists():
                out_path = output_dir / f"{safe_name}_{counter}.png"
                counter += 1

            image.save(out_path)
            records.append({
                "asset_name": asset_name,
                "type": obj.type.name,
                "file": out_path.name,
                "width": image.width,
                "height": image.height,
            })
        except Exception as exc:
            print(f"  [WARN] Could not extract '{asset_name}': {exc}")

    return records


def process_path(input_path: Path, output_dir: Path, name_filter: str | None) -> list[dict]:
    all_records = []
    suffix = input_path.suffix.lower()

    if suffix in (".apk", ".obb", ".zip"):
        print(f"Opening archive: {input_path.name}")
        with zipfile.ZipFile(input_path, "r") as zf:
            bundle_names = [n for n in zf.namelist() if "assets" in n.lower()]
            print(f"  Found {len(bundle_names)} asset entries inside archive")
            for bundle_name in bundle_names:
                try:
                    data = zf.read(bundle_name)
                    env = UnityPy.load(data)
                    records = extract_from_bundle(env, output_dir, name_filter)
                    if records:
                        print(f"  Extracted {len(records):4d} sprites from {bundle_name}")
                    all_records.extend(records)
                except Exception as exc:
                    print(f"  [WARN] Skipping {bundle_name}: {exc}")

    elif input_path.is_dir():
        bundle_files = list(input_path.rglob("*"))
        bundle_files = [f for f in bundle_files if f.is_file() and f.suffix not in (".py", ".txt", ".csv")]
        print(f"Scanning directory: {input_path} ({len(bundle_files)} files)")
        for bundle_file in bundle_files:
            try:
                env = UnityPy.load(str(bundle_file))
                records = extract_from_bundle(env, output_dir, name_filter)
                if records:
                    print(f"  Extracted {len(records):4d} sprites from {bundle_file.name}")
                all_records.extend(records)
            except Exception:
                pass  # Most files in a dir won't be Unity bundles

    else:
        print(f"Loading asset bundle: {input_path.name}")
        env = UnityPy.load(str(input_path))
        all_records = extract_from_bundle(env, output_dir, name_filter)

    return all_records


def write_summary(records: list[dict], output_dir: Path):
    summary_path = output_dir / "_extraction_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["asset_name", "type", "file", "width", "height"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSummary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract sprites from Rush Royale APK or Unity asset bundles")
    parser.add_argument("input", help="Path to .apk, .obb, asset bundle file, or directory")
    parser.add_argument("--filter", metavar="TEXT", help="Only extract assets whose name contains TEXT", default=None)
    parser.add_argument("--output", metavar="DIR", help="Output directory (default: assets/raw/)", default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"Input path does not exist: {input_path}")

    output_dir = Path(args.output).resolve() if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    if args.filter:
        print(f"Name filter: '{args.filter}'")
    print()

    records = process_path(input_path, output_dir, args.filter)

    if not records:
        print("\nNo sprites extracted. The APK may store assets in a downloaded OBB or runtime bundle.")
        print("Try pulling the app data directory from your device:")
        print("  adb pull /sdcard/Android/data/com.my.rush.royale/ ./rushroyal_data/")
        print("Then run: python tools/extract_assets.py ./rushroyal_data/")
    else:
        write_summary(records, output_dir)
        sprites = sum(1 for r in records if r["type"] == "Sprite")
        textures = sum(1 for r in records if r["type"] == "Texture2D")
        print(f"\nDone. Extracted {len(records)} assets ({sprites} sprites, {textures} textures)")


if __name__ == "__main__":
    main()
