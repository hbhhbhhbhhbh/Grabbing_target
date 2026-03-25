"""
Prepare and augment Pocari dataset for YOLO training.
All paths are relative to this bundle folder.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
BASE_DIR = Path(__file__).resolve().parent


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
    c = size // 2
    dx = int((size // 2) * np.cos(angle))
    dy = int((size // 2) * np.sin(angle))
    cv2.line(kernel, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_non_geometric_augment(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    if random.random() < 0.7:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.75, 1.25), beta=random.randint(-28, 28))
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-25, 25), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-20, 20), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.35:
        img = cv2.GaussianBlur(img, random.choice([(3, 3), (5, 5)]), sigmaX=0.8)
    if random.random() < 0.35:
        k = random.choice([5, 7, 9])
        img = cv2.filter2D(img, -1, build_motion_kernel(k, random.uniform(0.0, 2.0 * np.pi)))
    if random.random() < 0.35:
        sigma = random.uniform(4.0, 14.0)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        h, w = img.shape[:2]
        hole_w = random.randint(max(8, w // 20), max(12, w // 7))
        hole_h = random.randint(max(8, h // 20), max(12, h // 7))
        x1 = random.randint(0, max(0, w - hole_w))
        y1 = random.randint(0, max(0, h - hole_h))
        fill = int(random.uniform(20, 235))
        img[y1 : y1 + hole_h, x1 : x1 + hole_w] = (fill, fill, fill)
    return img


def apply_horizontal_flip(image: np.ndarray, boxes: List[YoloBox]) -> Tuple[np.ndarray, List[YoloBox]]:
    return cv2.flip(image, 1), [YoloBox(b.cls_id, 1.0 - b.x, b.y, b.w, b.h) for b in boxes]


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


def filter_boxes(boxes: List[YoloBox], target_class: int | None) -> List[YoloBox]:
    if target_class is None:
        return boxes
    return [YoloBox(0, b.x, b.y, b.w, b.h) for b in boxes if b.cls_id == target_class]


def write_sample(split: str, image: np.ndarray, boxes: List[YoloBox], out_root: Path, stem: str) -> None:
    cv2.imwrite(str(out_root / "images" / split / f"{stem}.jpg"), image)
    (out_root / "labels" / split / f"{stem}.txt").write_text(serialize_boxes(boxes), encoding="utf-8")


def write_yaml(output_root: Path, names: List[str]) -> None:
    (output_root / "dataset.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: yolo_dataset/images/train",
                "val: yolo_dataset/images/val",
                "test: yolo_dataset/images/test",
                f"nc: {len(names)}",
                "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default="raw/images")
    parser.add_argument("--labels-dir", type=str, default="raw/labels")
    parser.add_argument("--output-dir", type=str, default="yolo_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-aug", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--class-names", type=str, default="pocari,other")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = (BASE_DIR / args.images_dir).resolve()
    labels_dir = (BASE_DIR / args.labels_dir).resolve()
    out_root = (BASE_DIR / args.output_dir).resolve()

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    pairs = collect_pairs(images_dir, labels_dir)
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    split_map = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, items in split_map.items():
        for img_path, label_path in items:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            boxes = filter_boxes(parse_label_file(label_path), args.target_class)
            if not boxes:
                continue
            write_sample(split, img, boxes, out_root, img_path.stem)
            if split == "train":
                for i in range(args.num_aug):
                    aug_img = apply_non_geometric_augment(img)
                    aug_boxes = boxes
                    if random.random() < 0.5:
                        aug_img, aug_boxes = apply_horizontal_flip(aug_img, aug_boxes)
                    write_sample(split, aug_img, aug_boxes, out_root, f"{img_path.stem}_aug{i:02d}")

    raw_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if args.target_class is None:
        names = raw_names if raw_names else ["pocari", "other"]
    else:
        idx = args.target_class if raw_names and args.target_class < len(raw_names) else 0
        names = [raw_names[idx] if raw_names else "pocari"]
    write_yaml(BASE_DIR, names)
    print("Dataset prepared at:", out_root)
    print("YAML:", BASE_DIR / "dataset.yaml")


if __name__ == "__main__":
    main()
