"""
Build the talent icon reference library from data/screenshots/Talent N Icon files.

Each tier badge looks the same regardless of branch (L/R/Fixed) — the game only
shows which tier is active on the board, not which branch was chosen. So every
branch variant for a given tier gets a copy of the same badge image.

Branch detection at runtime therefore relies on the MCR's observation history,
not visual badge distinction. The classifier's role is tier detection only.

Output:
    assets/reference/talent_icons/
        1_L.png  1_R.png  1_Fixed.png
        2_L.png  2_R.png  2_Fixed.png
        3_L.png  3_R.png  3_Fixed.png
        4_L.png  4_R.png  4_Fixed.png

Run from the project root:
    python tools/build_talent_icon_library.py
"""

from pathlib import Path
from PIL import Image

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SCREENSHOTS   = PROJECT_ROOT / "data" / "screenshots"
ICON_REF_DIR  = PROJECT_ROOT / "assets" / "reference" / "talent_icons"

BRANCHES = ("L", "R", "Fixed")

# Map tier -> source filename (with extension)
SOURCES = {
    1: "Talent 1 Icon.png",
    2: "Talent 2 Icon.png",
    3: "Talent 3 Icon.jpg",
    4: "Talent 4 Icon.jpg",
}

# For tiers whose source image has surrounding context, crop to just the badge.
# Values are (left%, top%, right%, bottom%) as fractions of the image dimensions.
# Set to None to use the full image as-is.
CROP_FRACTIONS = {
    1: None,
    2: None,
    3: (0.20, 0.10, 0.80, 0.88),   # trim reticle/crosshair surround
    4: (0.00, 0.05, 0.85, 0.95),   # trim green edge strip
}


def build():
    ICON_REF_DIR.mkdir(parents=True, exist_ok=True)

    for tier, filename in SOURCES.items():
        src = SCREENSHOTS / filename
        if not src.exists():
            print(f"  [missing] {src} — skipping tier {tier}")
            continue

        img = Image.open(src).convert("RGB")

        crop = CROP_FRACTIONS[tier]
        if crop is not None:
            w, h = img.size
            box = (
                int(crop[0] * w),
                int(crop[1] * h),
                int(crop[2] * w),
                int(crop[3] * h),
            )
            img = img.crop(box)

        for branch in BRANCHES:
            dest = ICON_REF_DIR / f"{tier}_{branch}.png"
            img.save(dest, "PNG")

        print(f"  Tier {tier} ({filename}) -> {tier}_L/R/Fixed.png  {img.size}")

    print(f"\nDone. {len(SOURCES) * 3} files written to {ICON_REF_DIR}")
    print("\nNote: L/R/Fixed variants are identical copies per tier.")
    print("Branch detection relies on MCR observation history, not badge visuals.")


if __name__ == "__main__":
    build()