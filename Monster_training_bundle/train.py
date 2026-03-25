"""
Portable YOLO training entrypoint.
Run this from any location after moving the whole folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO model from bundle dataset.")
    parser.add_argument("--data", type=str, default="dataset.yaml")
    parser.add_argument("--model", type=str, default="weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="pocari_special")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_path = (BASE_DIR / args.data).resolve()
    model_path = (BASE_DIR / args.model).resolve()
    project_path = (BASE_DIR / args.project).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml missing: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        device=args.device,
        pretrained=True,
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

    best = project_path / args.name / "weights" / "best.pt"
    if best.exists():
        print("Best checkpoint:", best)
        tuned = YOLO(str(best))
        metrics = tuned.val(data=str(data_path), split="test", imgsz=args.imgsz, device=args.device)
        print("Test metrics:", metrics)
    else:
        print("Training finished but best.pt not found.")


if __name__ == "__main__":
    main()
