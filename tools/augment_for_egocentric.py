"""
第一视角数据增强工具 - 针对视障辅助抓取场景

功能:
1. 运动模糊注入：模拟相机抖动产生的拖影
2. 随机遮挡 (Cutout): 强迫模型学习局部特征
3. 批量处理脚本：支持对自有数据集的预处理

使用示例:
    python augment_for_egocentric.py --input data/images --output data/augmented \
        --motion-blur --cutout --num-aug 5
"""

import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional
import json


def create_motion_kernel(
    size: int,
    angle: float,
    intensity: float = 1.0
) -> np.ndarray:
    """
    创建运动模糊核
    
    Args:
        size: 模糊核大小
        angle: 运动方向角度 (弧度)
        intensity: 强度因子
    
    Returns:
        运动模糊核矩阵
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    
    # 计算运动轨迹
    center = size // 2
    length = int(size * intensity)
    
    x1 = center - int(length * np.cos(angle) / 2)
    y1 = center - int(length * np.sin(angle) / 2)
    x2 = center + int(length * np.cos(angle) / 2)
    y2 = center + int(length * np.sin(angle) / 2)
    
    # 确保在核范围内
    x1, x2 = max(0, x1), min(size - 1, x2)
    y1, y2 = max(0, y1), min(size - 1, y2)
    
    # 画线生成运动轨迹
    if x1 != x2 or y1 != y2:
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
    
    # 归一化
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    
    return kernel


def apply_motion_blur(
    image: np.ndarray,
    max_kernel_size: int = 15,
    min_kernel_size: int = 3
) -> np.ndarray:
    """
    应用运动模糊
    
    Args:
        image: 输入图像
        max_kernel_size: 最大模糊核大小
        min_kernel_size: 最小模糊核大小
    
    Returns:
        运动模糊后的图像
    """
    # 随机选择模糊核大小和方向
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    direction = random.uniform(0, 2 * np.pi)
    intensity = random.uniform(0.5, 1.0)
    
    # 创建运动模糊核
    kernel = create_motion_kernel(kernel_size, direction, intensity)
    
    # 应用模糊
    blurred = cv2.filter2D(image, -1, kernel)
    
    # 随机混合原图和模糊图 (模拟不同程度的模糊)
    blend_ratio = random.uniform(0.3, 0.8)
    result = cv2.addWeighted(image, 1 - blend_ratio, blurred, blend_ratio, 0)
    
    return result


def apply_cutout(
    image: np.ndarray,
    n_holes: int = 3,
    max_h_size: int = 50,
    min_h_size: int = 10,
    fill_value: int = 0
) -> np.ndarray:
    """
    应用 Cutout 随机遮挡
    
    Args:
        image: 输入图像
        n_holes: 遮挡区域数量
        max_h_size: 最大遮挡区域尺寸
        min_h_size: 最小遮挡区域尺寸
        fill_value: 填充值 (0=黑色，127=灰色，或使用图像均值)
    
    Returns:
        遮挡后的图像
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    for _ in range(n_holes):
        # 随机生成遮挡区域
        h_size = random.randint(min_h_size, max_h_size)
        w_size = random.randint(min_h_size, max_h_size)
        
        y = random.randint(0, h)
        x = random.randint(0, w)
        
        # 计算遮挡区域边界
        y1 = max(0, y - h_size // 2)
        y2 = min(h, y + h_size // 2)
        x1 = max(0, x - w_size // 2)
        x2 = min(w, x + w_size // 2)
        
        # 确定填充值
        if fill_value == 'mean':
            # 使用图像均值填充
            fill_color = [int(result[y1:y2, x1:x2].mean())] * 3
        else:
            fill_color = [fill_value] * 3
        
        # 应用遮挡
        result[y1:y2, x1:x2] = fill_color
    
    return result


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 25.0
) -> np.ndarray:
    """
    添加高斯噪声
    
    Args:
        image: 输入图像
        sigma: 噪声标准差
    
    Returns:
        添加噪声后的图像
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    return noisy_image


def apply_brightness_adjustment(
    image: np.ndarray,
    alpha: Optional[float] = None,
    beta: Optional[int] = None
) -> np.ndarray:
    """
    调整亮度和对比度
    
    Args:
        image: 输入图像
        alpha: 对比度增益 (默认随机 0.5-1.5)
        beta: 亮度偏移 (默认随机 -50 到 50)
    
    Returns:
        调整后的图像
    """
    if alpha is None:
        alpha = random.uniform(0.5, 1.5)
    if beta is None:
        beta = random.randint(-50, 50)
    
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def apply_random_rotation(
    image: np.ndarray,
    max_angle: float = 30.0
) -> np.ndarray:
    """
    随机旋转图像
    
    Args:
        image: 输入图像
        max_angle: 最大旋转角度
    
    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-max_angle, max_angle)
    
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), 
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated


def apply_compose_augmentation(
    image: np.ndarray,
    aug_config: dict
) -> np.ndarray:
    """
    应用组合增强
    
    Args:
        image: 输入图像
        aug_config: 增强配置字典
    
    Returns:
        增强后的图像
    """
    result = image.copy()
    
    # 运动模糊
    if aug_config.get('motion_blur', False):
        prob = aug_config.get('motion_blur_prob', 0.5)
        if random.random() < prob:
            max_kernel = aug_config.get('max_kernel_size', 15)
            result = apply_motion_blur(result, max_kernel_size=max_kernel)
    
    # Cutout 遮挡
    if aug_config.get('cutout', False):
        prob = aug_config.get('cutout_prob', 0.5)
        if random.random() < prob:
            n_holes = aug_config.get('n_holes', 3)
            max_h_size = aug_config.get('max_h_size', 50)
            result = apply_cutout(result, n_holes=n_holes, max_h_size=max_h_size)
    
    # 高斯噪声
    if aug_config.get('noise', False):
        prob = aug_config.get('noise_prob', 0.3)
        if random.random() < prob:
            sigma = aug_config.get('noise_sigma', 25)
            result = apply_gaussian_noise(result, sigma=sigma)
    
    # 亮度调整
    if aug_config.get('brightness', False):
        prob = aug_config.get('brightness_prob', 0.5)
        if random.random() < prob:
            result = apply_brightness_adjustment(result)
    
    # 随机旋转
    if aug_config.get('rotation', False):
        prob = aug_config.get('rotation_prob', 0.3)
        if random.random() < prob:
            max_angle = aug_config.get('max_rotation_angle', 30)
            result = apply_random_rotation(result, max_angle=max_angle)
    
    return result


def process_dataset(
    input_dir: str,
    output_dir: str,
    num_aug_per_image: int = 5,
    aug_config: Optional[dict] = None,
    save_annotations: bool = True
):
    """
    批量处理数据集
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出增强图像目录
        num_aug_per_image: 每张原始图像的增强版本数量
        aug_config: 增强配置
        save_annotations: 是否保存标注文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if aug_config is None:
        aug_config = {
            'motion_blur': True,
            'motion_blur_prob': 0.6,
            'max_kernel_size': 15,
            'cutout': True,
            'cutout_prob': 0.7,
            'n_holes': 3,
            'max_h_size': 50,
            'noise': True,
            'noise_prob': 0.3,
            'brightness': True,
            'brightness_prob': 0.5,
            'rotation': True,
            'rotation_prob': 0.3,
            'max_rotation_angle': 30
        }
    
    # 查找所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    annotations = []
    
    for idx, img_file in enumerate(image_files):
        print(f"Processing [{idx + 1}/{len(image_files)}]: {img_file.name}")
        
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  Warning: Could not read {img_file.name}, skipping...")
            continue
        
        # 保存原始图像
        base_name = img_file.stem
        ext = img_file.suffix
        output_img_path = output_path / f"{base_name}{ext}"
        cv2.imwrite(str(output_img_path), image)
        
        # 记录原始标注
        if save_annotations:
            annotations.append({
                'filename': f"{base_name}{ext}",
                'original': str(img_file.name),
                'augmented': False,
                'aug_config': None
            })
        
        # 生成增强版本
        for aug_idx in range(num_aug_per_image):
            # 应用组合增强
            augmented = apply_compose_augmentation(image, aug_config)
            
            # 保存增强图像
            aug_filename = f"{base_name}_aug{aug_idx:02d}{ext}"
            aug_img_path = output_path / aug_filename
            cv2.imwrite(str(aug_img_path), augmented)
            
            # 记录增强标注
            if save_annotations:
                annotations.append({
                    'filename': aug_filename,
                    'original': str(img_file.name),
                    'augmented': True,
                    'aug_idx': aug_idx,
                    'aug_config': aug_config
                })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1} images, generated {(idx + 1) * num_aug_per_image} augmented versions")
    
    # 保存标注信息
    if save_annotations and annotations:
        anno_file = output_path / 'augmentation_annotations.json'
        with open(anno_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        print(f"\nSaved augmentation annotations to {anno_file}")
    
    print(f"\nCompleted! Generated {len(annotations)} total images")


def main():
    parser = argparse.ArgumentParser(
        description='第一视角数据增强工具 - 针对视障辅助抓取场景'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入图像目录'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出增强图像目录'
    )
    parser.add_argument(
        '--num-aug', '-n',
        type=int,
        default=5,
        help='每张原始图像的增强版本数量 (默认：5)'
    )
    parser.add_argument(
        '--motion-blur',
        action='store_true',
        help='启用运动模糊增强'
    )
    parser.add_argument(
        '--cutout',
        action='store_true',
        help='启用 Cutout 遮挡增强'
    )
    parser.add_argument(
        '--noise',
        action='store_true',
        help='启用高斯噪声增强'
    )
    parser.add_argument(
        '--brightness',
        action='store_true',
        help='启用亮度调整增强'
    )
    parser.add_argument(
        '--rotation',
        action='store_true',
        help='启用随机旋转增强'
    )
    parser.add_argument(
        '--no-annotations',
        action='store_true',
        help='不保存标注文件'
    )
    
    args = parser.parse_args()
    
    # 构建增强配置
    aug_config = {
        'motion_blur': args.motion_blur,
        'motion_blur_prob': 0.6,
        'max_kernel_size': 15,
        'cutout': args.cutout,
        'cutout_prob': 0.7,
        'n_holes': 3,
        'max_h_size': 50,
        'noise': args.noise,
        'noise_prob': 0.3,
        'brightness': args.brightness,
        'brightness_prob': 0.5,
        'rotation': args.rotation,
        'rotation_prob': 0.3,
        'max_rotation_angle': 30
    }
    
    # 如果没有启用任何增强，使用默认配置
    if not any([args.motion_blur, args.cutout, args.noise, 
                args.brightness, args.rotation]):
        print("No augmentation methods specified, using default config...")
        aug_config = {
            'motion_blur': True,
            'cutout': True,
            'noise': True,
            'brightness': True,
            'rotation': True
        }
    
    print("=" * 60)
    print("第一视角数据增强工具")
    print("=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Num augmentations per image: {args.num_aug}")
    print("\nEnabled augmentations:")
    for key, value in aug_config.items():
        if isinstance(value, bool) and value:
            print(f"  ✓ {key}")
    print("=" * 60)
    
    # 处理数据集
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_aug_per_image=args.num_aug,
        aug_config=aug_config,
        save_annotations=not args.no_annotations
    )


if __name__ == '__main__':
    main()
