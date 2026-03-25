"""
Train a custom YOLO model for Pocari dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom YOLO model for Pocari object detection.")
    parser.add_argument("--data", type=str, default="datasets/pocari_yolo_aug/dataset.yaml", help="Dataset yaml path")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/pocari_special")
    parser.add_argument("--name", type=str, default="yolo11n_ft")
    parser.add_argument("--device", type=str, default="0", help="GPU id or 'cpu'")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=True,
        cache=False,
        val=True,
        save=True,
        plots=True,
        degrees=3.0,
        translate=0.06,
        scale=0.18,
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.05,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.35,
    )

    best_weight = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weight.exists():
        print(f"Best checkpoint: {best_weight}")
        tuned_model = YOLO(str(best_weight))
        metrics = tuned_model.val(data=str(data_path), split="test", imgsz=args.imgsz, device=args.device)
        print("Test metrics:")
        print(metrics)
    else:
        print("Training finished but best.pt was not found.")


if __name__ == "__main__":
    main()
