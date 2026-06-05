"""
Train a YOLOv8 classification model on board cell unit crops.

Expects the dataset built by tools/prepare_dataset.py to exist at
datasets/unit_classifier/ (or the path given via --data).

The trained model is saved to assets/models/unit_classifier.pt.

Workflow:
    # 1. Build the dataset (run once, or when new reference images are added)
    python tools/prepare_dataset.py

    # 2. Train
    python tools/train_classifier.py [options]

    # 3. Evaluate (optional — prints val accuracy)
    python tools/train_classifier.py --eval-only --model assets/models/unit_classifier.pt

Key options:
    --model    yolov8n-cls.pt   Base model weight (nano = default, fast CPU inference)
    --epochs   100              Training epochs
    --imgsz    128              Input image size (must match prepare_dataset output)
    --batch    32               Batch size
    --device   cpu              cpu | 0 | 0,1  (GPU index or 'cpu')
    --workers  0                DataLoader workers (0 = main thread, safe on Windows)
    --patience 20               Early-stopping patience (0 = disable)

Notes:
  - With only rank-1 reference images (~81 units × 21 images = ~1700 total after
    augmentation), expect ~80-90% top-1 val accuracy on rank-1 cells.
  - Accuracy on rank 5-7 cells will be low until rank-specific images are added.
  - Re-run prepare_dataset.py then this script whenever new images are added.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA   = PROJECT_ROOT / "datasets" / "unit_classifier"
DEFAULT_OUT    = PROJECT_ROOT / "assets"   / "models"
DEFAULT_MODEL  = "yolov8n-cls.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ  = 128
DEFAULT_BATCH  = 32


def train(data: Path,
          out_dir: Path,
          base_model: str,
          epochs: int,
          imgsz: int,
          batch: int,
          device: str,
          workers: int,
          patience: int) -> Path:
    """
    Run YOLOv8 classification training and copy the best weights to out_dir.

    Returns:
        Path to the saved model file (assets/models/unit_classifier.pt).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.  Run: pip install ultralytics",
              file=sys.stderr)
        sys.exit(1)

    if not data.is_dir():
        print(f"ERROR: Dataset not found at {data}\n"
              f"       Run tools/prepare_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    # Count classes so we can sanity-check the dataset
    train_dir = data / "train"
    classes = [d.name for d in sorted(train_dir.iterdir()) if d.is_dir()] \
              if train_dir.is_dir() else []
    if not classes:
        print(f"ERROR: No class sub-directories found in {train_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset     : {data}")
    print(f"Classes     : {len(classes)} units")
    print(f"Base model  : {base_model}")
    print(f"Epochs      : {epochs}   imgsz={imgsz}   batch={batch}   device={device}")
    print()

    model = YOLO(base_model)

    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        # Augmentation — YOLOv8 built-ins complement prepare_dataset.py augments
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.15,
        # Project / naming
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name="unit_classifier",
        exist_ok=True,
        verbose=True,
    )

    # YOLOv8 saves best weights to runs/classify/unit_classifier/weights/best.pt
    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"WARNING: Expected best.pt at {best_weights} — not found.",
              file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "unit_classifier.pt"
    shutil.copy2(best_weights, dest)
    print(f"\nModel saved to {dest}")
    return dest


def evaluate(model_path: Path, data: Path, imgsz: int, device: str) -> None:
    """Run validation on the saved model and print top-1/top-5 accuracy."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data), imgsz=imgsz, device=device, verbose=True)
    top1 = getattr(metrics, "top1", None)
    top5 = getattr(metrics, "top5", None)
    if top1 is not None:
        print(f"\nTop-1 accuracy : {top1:.4f}")
    if top5 is not None:
        print(f"Top-5 accuracy : {top5:.4f}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8 classification model on board cell crops."
    )
    p.add_argument("--data",       default=str(DEFAULT_DATA),
                   help="Dataset root (default: datasets/unit_classifier)")
    p.add_argument("--out",        default=str(DEFAULT_OUT),
                   help="Output dir for saved model (default: assets/models)")
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help="Base YOLO model (default: yolov8n-cls.pt)")
    p.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",      type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch",      type=int, default=DEFAULT_BATCH)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--workers",    type=int, default=0,
                   help="DataLoader workers (default: 0, safe on Windows)")
    p.add_argument("--patience",   type=int, default=20,
                   help="Early-stopping patience, 0 = disabled (default: 20)")
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training; evaluate existing model only")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_path  = Path(args.data)
    out_path   = Path(args.out)

    if args.eval_only:
        model_path = out_path / "unit_classifier.pt"
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
            sys.exit(1)
        evaluate(model_path, data_path, args.imgsz, args.device)
    else:
        train(
            data       = data_path,
            out_dir    = out_path,
            base_model = args.model,
            epochs     = args.epochs,
            imgsz      = args.imgsz,
            batch      = args.batch,
            device     = args.device,
            workers    = args.workers,
            patience   = args.patience,
        )