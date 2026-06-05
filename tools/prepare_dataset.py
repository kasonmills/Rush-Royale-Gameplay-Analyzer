"""
Build the YOLOv8 classification dataset for unit_classifier training.

Reads every PNG from assets/reference/<unit_id>/ and generates synthetic
augmented variants so the model can generalise beyond the single source
image typically available per unit.

Output layout (YOLOv8 classification directory format):
    datasets/unit_classifier/
        train/
            alchemist/
                base_rank1.png        ← original (train split)
                aug_001.png
                ...
            archer/
                ...
        val/
            alchemist/
                aug_N.png             ← held-out augmentations
            ...

Adding new units or ranks
--------------------------
Add the image to assets/reference/<unit_id>/ and re-run this script.
No code changes required.

Augmentations applied (using only Pillow + numpy — no extra dependencies):
    • Horizontal flip
    • Rotation  ±10° and ±20°
    • Brightness  −25% / +25%
    • Contrast    −20% / +20%
    • Saturation  +40%  (simulates buff glows)
    • Scale-crop   85% and 115% (simulates zoom / perspective variation)
    • Gaussian noise  σ=10 and σ=20  (simulates video compression)
    • Combinations of the above

Run from project root:
    python tools/prepare_dataset.py [--ref assets/reference] [--out datasets/unit_classifier]
                                    [--val-frac 0.2] [--n-aug 20] [--seed 42]
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Augmentation helpers (Pillow + numpy only)
# ---------------------------------------------------------------------------

def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1])  # BGR → RGB


def _to_arr(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB → BGR


def aug_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def aug_rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def aug_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def aug_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def aug_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def aug_scale_crop(img: Image.Image, scale: float) -> Image.Image:
    """Scale then centre-crop back to original size."""
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    if scale > 1.0:
        left = (new_w - w) // 2
        top  = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        canvas.paste(resized, (x, y))
        return canvas


def aug_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ---------------------------------------------------------------------------
# Augmentation catalogue
# Each entry is a callable (img, rng) → img
# ---------------------------------------------------------------------------

def _build_augment_fns(rng: np.random.Generator):
    return [
        lambda img, _: aug_flip(img),
        lambda img, _: aug_rotate(img, 10),
        lambda img, _: aug_rotate(img, -10),
        lambda img, _: aug_rotate(img, 20),
        lambda img, _: aug_rotate(img, -20),
        lambda img, _: aug_brightness(img, 0.75),
        lambda img, _: aug_brightness(img, 1.25),
        lambda img, _: aug_contrast(img, 0.80),
        lambda img, _: aug_contrast(img, 1.20),
        lambda img, _: aug_saturation(img, 1.40),
        lambda img, _: aug_scale_crop(img, 0.85),
        lambda img, _: aug_scale_crop(img, 1.15),
        lambda img, r: aug_noise(img, 10, r),
        lambda img, r: aug_noise(img, 20, r),
        # Combinations
        lambda img, _: aug_brightness(aug_flip(img), 1.20),
        lambda img, _: aug_rotate(aug_flip(img), 10),
        lambda img, _: aug_brightness(aug_rotate(img, 15), 0.80),
        lambda img, _: aug_contrast(aug_scale_crop(img, 1.10), 1.15),
        lambda img, r: aug_noise(aug_brightness(img, 0.90), 15, r),
        lambda img, r: aug_noise(aug_flip(aug_scale_crop(img, 0.90)), 10, r),
    ]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(ref_dir: Path,
                  out_dir: Path,
                  val_frac: float = 0.2,
                  n_aug: int = 20,
                  seed: int = 42) -> None:
    """
    Scan ref_dir for unit sub-directories and build the dataset in out_dir.

    Args:
        ref_dir:   assets/reference/ root (contains <unit_id>/ subdirs).
        out_dir:   Output root (datasets/unit_classifier/).
        val_frac:  Fraction of augmented images to put into val/ split.
        n_aug:     Number of augmented images to generate per source image.
        seed:      Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    if out_dir.exists():
        print(f"Removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)

    train_root = out_dir / "train"
    val_root   = out_dir / "val"

    unit_dirs = sorted(d for d in ref_dir.iterdir() if d.is_dir())
    if not unit_dirs:
        print(f"ERROR: No unit directories found in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    aug_fns = _build_augment_fns(rng)
    # Cycle through augmentation functions if n_aug > len(aug_fns)
    selected_fns = [aug_fns[i % len(aug_fns)] for i in range(n_aug)]

    total_train = 0
    total_val   = 0

    for unit_dir in unit_dirs:
        unit_id = unit_dir.name
        source_images = sorted(unit_dir.glob("*.png"))
        if not source_images:
            print(f"  SKIP {unit_id} — no PNG images found")
            continue

        train_class = train_root / unit_id
        val_class   = val_root   / unit_id
        train_class.mkdir(parents=True, exist_ok=True)
        val_class.mkdir(parents=True,   exist_ok=True)

        aug_idx = 0
        for src_path in source_images:
            img = Image.open(src_path).convert("RGB")

            # Original always goes to train
            out_path = train_class / src_path.name
            img.save(out_path)
            total_train += 1

            # Generate augmented variants
            all_aug: list[tuple[str, Image.Image]] = []
            for fn in selected_fns:
                aug_img = fn(img, rng)
                all_aug.append((f"aug_{aug_idx:04d}.png", aug_img))
                aug_idx += 1

            # Split augmented images into train/val
            n_val = max(1, int(len(all_aug) * val_frac))
            val_indices = set(py_rng.sample(range(len(all_aug)), n_val))

            for j, (fname, aug_img) in enumerate(all_aug):
                dest = val_class / fname if j in val_indices else train_class / fname
                aug_img.save(dest)
                if j in val_indices:
                    total_val += 1
                else:
                    total_train += 1

        print(f"  {unit_id:30s}  sources={len(source_images)}")

    print(f"\nDataset written to {out_dir}")
    print(f"  Classes  : {len(unit_dirs)}")
    print(f"  Train    : {total_train} images")
    print(f"  Val      : {total_val} images")
    print(f"  Total    : {total_train + total_val} images")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build YOLOv8 classification dataset from reference images."
    )
    p.add_argument("--ref",      default="assets/reference",
                   help="Reference image root (default: assets/reference)")
    p.add_argument("--out",      default="datasets/unit_classifier",
                   help="Output dataset root (default: datasets/unit_classifier)")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of augmented images for validation (default: 0.2)")
    p.add_argument("--n-aug",    type=int,   default=20,
                   help="Augmented images per source image (default: 20)")
    p.add_argument("--seed",     type=int,   default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ref_dir = PROJECT_ROOT / args.ref
    out_dir = PROJECT_ROOT / args.out

    print(f"Reference dir : {ref_dir}")
    print(f"Output dir    : {out_dir}")
    print(f"Val fraction  : {args.val_frac}")
    print(f"Augments/src  : {args.n_aug}")
    print(f"Seed          : {args.seed}")
    print()

    build_dataset(ref_dir, out_dir, args.val_frac, args.n_aug, args.seed)