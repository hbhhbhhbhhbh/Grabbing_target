"""
Prepare and augment Pocari dataset for YOLO detection.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class YoloBox:
    cls_id: int
    x: float
    y: float
    w: float
    h: float


def parse_label_file(label_path: Path) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        boxes.append(YoloBox(cls_id=cls_id, x=x, y=y, w=w, h=h))
    return boxes


def serialize_boxes(boxes: List[YoloBox]) -> str:
    return "".join(f"{b.cls_id} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


def build_motion_kernel(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    dx = int((size // 2) * np.cos(angle))
    dy = int((size // 2) * np.sin(angle))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_non_geometric_augment(image: np.ndarray) -> np.ndarray:
    """Apply photometric augmentations that do not change bbox coordinates."""
    img = image.copy()

    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta = random.randint(-28, 28)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-25, 25), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-20, 20), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0.8)

    if random.random() < 0.35:
        k = random.choice([5, 7, 9])
        angle = random.uniform(0.0, 2.0 * np.pi)
        img = cv2.filter2D(img, -1, build_motion_kernel(k, angle))

    if random.random() < 0.35:
        sigma = random.uniform(4.0, 14.0)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        h, w = img.shape[:2]
        for _ in range(random.randint(1, 2)):
            hole_w = random.randint(max(8, w // 20), max(12, w // 7))
            hole_h = random.randint(max(8, h // 20), max(12, h // 7))
            x1 = random.randint(0, max(0, w - hole_w))
            y1 = random.randint(0, max(0, h - hole_h))
            fill = int(random.uniform(20, 235))
            img[y1 : y1 + hole_h, x1 : x1 + hole_w] = (fill, fill, fill)

    return img


def apply_horizontal_flip(image: np.ndarray, boxes: List[YoloBox]) -> Tuple[np.ndarray, List[YoloBox]]:
    flipped = cv2.flip(image, 1)
    new_boxes = [YoloBox(cls_id=b.cls_id, x=1.0 - b.x, y=b.y, w=b.w, h=b.h) for b in boxes]
    return flipped, new_boxes


def write_dataset_yaml(output_root: Path, class_names: List[str]) -> Path:
    yaml_path = output_root / "dataset.yaml"
    names_text = ", ".join([f"'{name}'" for name in class_names])
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(class_names)}",
                f"names: [{names_text}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def ensure_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def filter_and_remap_boxes(boxes: List[YoloBox], target_class: int | None) -> List[YoloBox]:
    if target_class is None:
        return boxes
    kept = [b for b in boxes if b.cls_id == target_class]
    return [YoloBox(cls_id=0, x=b.x, y=b.y, w=b.w, h=b.h) for b in kept]


def write_sample(
    split: str,
    image: np.ndarray,
    boxes: List[YoloBox],
    out_root: Path,
    out_stem: str,
) -> None:
    cv2.imwrite(str(out_root / "images" / split / f"{out_stem}.jpg"), image)
    (out_root / "labels" / split / f"{out_stem}.txt").write_text(serialize_boxes(boxes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and augment Pocari dataset for YOLO.")
    parser.add_argument("--images-dir", type=str, default="pocari_dataset")
    parser.add_argument("--labels-dir", type=str, default="Labels")
    parser.add_argument("--output-dir", type=str, default="datasets/pocari_yolo_aug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-aug", type=int, default=3, help="Augmented copies per train image")
    parser.add_argument("--target-class", type=int, default=None, help="Keep only this class and remap to 0")
    parser.add_argument("--class-names", type=str, default="pocari,other", help="Comma-separated class names")
    parser.add_argument("--force", action="store_true", help="Delete output directory first")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_root = Path(args.output_dir)

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No valid image/label pairs found.")

    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    split_map = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    stats = {
        "total_pairs": n_total,
        "split_counts": {"train": len(split_map["train"]), "val": len(split_map["val"]), "test": len(split_map["test"])},
        "num_aug_per_train_image": args.num_aug,
        "target_class": args.target_class,
    }

    for split, items in split_map.items():
        for img_path, label_path in items:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            boxes = parse_label_file(label_path)
            boxes = filter_and_remap_boxes(boxes, args.target_class)
            if not boxes:
                continue

            base_stem = img_path.stem
            write_sample(split, image, boxes, out_root, base_stem)

            if split != "train":
                continue

            for aug_idx in range(args.num_aug):
                aug_img = apply_non_geometric_augment(image)
                aug_boxes = boxes
                if random.random() < 0.5:
                    aug_img, aug_boxes = apply_horizontal_flip(aug_img, aug_boxes)
                out_stem = f"{base_stem}_aug{aug_idx:02d}"
                write_sample("train", aug_img, aug_boxes, out_root, out_stem)

    raw_class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    class_names = raw_class_names
    if args.target_class is not None:
        pick = args.target_class if args.target_class < len(raw_class_names) else 0
        class_names = [raw_class_names[pick] if raw_class_names else "target_object"]

    yaml_path = write_dataset_yaml(out_root, class_names)
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("YOLO dataset prepared")
    print("=" * 60)
    print(f"Total pairs: {n_total}")
    print(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
    print(f"Aug per train image: {args.num_aug}")
    print(f"Target class mode: {args.target_class}")
    print(f"Dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
