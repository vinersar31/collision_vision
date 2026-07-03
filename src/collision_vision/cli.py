"""Command-line interface for CollisionVision.

Subcommands:
    preprocess  Convert VIA/COCO annotations to a YOLOv8-Seg dataset.
    train       Fine-tune a YOLOv8-Seg model from ``config.yaml``.
    infer       Run segmentation on an image or folder and save overlays.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .preprocess import DEFAULT_CLASSES, DatasetPreprocessor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collision-vision",
        description="Car-damage instance segmentation with YOLOv8-Seg.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preprocess --------------------------------------------------------
    pre = subparsers.add_parser("preprocess", help="Build a YOLOv8-Seg dataset from annotations.")
    pre.add_argument("--format", choices=["via", "coco"], required=True, help="Annotation format.")
    pre.add_argument("--annotations", required=True, help="Path to the annotation JSON export.")
    pre.add_argument("--images", required=True, help="Directory containing the source images.")
    pre.add_argument("--output", default="data/processed", help="Output dataset root.")
    pre.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction.")
    pre.add_argument("--seed", type=int, default=0, help="Random seed for the split.")
    pre.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="Ordered class names.")
    pre.add_argument("--attribute-key", default=None, help="VIA region_attributes key holding the label.")

    # --- train -------------------------------------------------------------
    train = subparsers.add_parser("train", help="Fine-tune a YOLOv8-Seg model.")
    train.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    train.add_argument("--model", default=None, help="Override the checkpoint (e.g. yolov8n-seg.pt).")
    train.add_argument("--epochs", type=int, default=None, help="Override the number of epochs.")
    train.add_argument("--batch", type=int, default=None, help="Override the batch size.")
    train.add_argument("--imgsz", type=int, default=None, help="Override the training image size.")
    train.add_argument("--device", default=None, help="Override the training device.")

    # --- infer -------------------------------------------------------------
    infer = subparsers.add_parser("infer", help="Segment images and save overlays.")
    infer.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    infer.add_argument("--source", required=True, help="Image file or directory of images.")
    infer.add_argument("--output", default="outputs", help="Directory to write overlays into.")
    infer.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")

    return parser


def _run_preprocess(args: argparse.Namespace) -> None:
    preprocessor = DatasetPreprocessor(class_names=args.classes)
    data_yaml = preprocessor.run(
        fmt=args.format,
        annotations=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        attribute_key=args.attribute_key,
    )
    print(f"Dataset ready. Descriptor: {data_yaml}")


def _run_train(args: argparse.Namespace) -> None:
    from .config import Config
    from .train import SegmentationTrainer

    config = Config.from_yaml(args.config)
    if args.model:
        config.model = args.model
    trainer = SegmentationTrainer(config)
    trainer.train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)


def _run_infer(args: argparse.Namespace) -> None:
    import cv2

    from .inference import DamageSegmenter
    from .visualize import MaskVisualizer

    segmenter = DamageSegmenter(weights=args.weights, conf=args.conf)
    visualizer = MaskVisualizer()

    source = Path(args.source)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)
    else:
        images = [source]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Could not read %s; skipping", image_path)
            continue
        instances = segmenter.predict(image)
        overlay = visualizer.overlay(image, instances)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), overlay)
        print(f"{image_path.name}: {len(instances)} damage region(s) -> {out_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    dispatch = {
        "preprocess": _run_preprocess,
        "train": _run_train,
        "infer": _run_infer,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
