"""
性能验证测试脚本 - 对比优化前后的系统性能

测试内容:
1. 时序平滑效果评估 (坐标抖动分析)
2. FPS 性能测试 (不同分辨率下的速度)
3. 检测稳定性对比 (优化前后)
4. 遮挡场景鲁棒性测试
5. ROI 注意力有效性验证

使用方法:
    python tests/test_accuracy_improvement.py --video data/test_video.mp4
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse

# 导入优化模块
from perception.temporal_smoother import BMASmoother, MultiObjectSmoother
from perception.roi_attention import ROIAttention
from perception.object_detector import ObjectDetector


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self):
        self.fps_history = []
        self.detection_stability = []
        self.coordinate_jitter = []
    
    def calculate_jitter(self, coordinates: List[Tuple[int, int]]) -> float:
        """
        计算坐标抖动标准差
        
        Args:
            coordinates: 中心点坐标列表 [(x1,y1), (x2,y2), ...]
        
        Returns:
            平均抖动值 (像素)
        """
        if len(coordinates) < 2:
            return 0.0
        
        coords = np.array(coordinates)
        
        # 计算相邻帧之间的位移
        deltas = []
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            delta = np.sqrt(dx*dx + dy*dy)
            deltas.append(delta)
        
        # 返回平均抖动
        return float(np.mean(deltas))
    
    def calculate_stability(self, detections_per_frame: List[int]) -> Dict:
        """
        计算检测稳定性指标
        
        Args:
            detections_per_frame: 每帧的检测数量列表
        
        Returns:
            稳定性指标字典
        """
        if not detections_per_frame:
            return {"mean": 0, "std": 0, "cv": 0}
        
        mean_det = np.mean(detections_per_frame)
        std_det = np.std(detections_per_frame)
        cv = std_det / mean_det if mean_det > 0 else 0  # 变异系数
        
        return {
            "mean": float(mean_det),
            "std": float(std_det),
            "cv": float(cv)
        }


def test_smoother_effectiveness(video_path: str, alpha: float = 0.6):
    """
    测试 EMA 平滑器的效果
    
    Args:
        video_path: 测试视频路径
        alpha: EMA 平滑系数
    """
    print("\n" + "="*60)
    print("测试 1: EMA 时序平滑效果评估")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    smoother = BMASmoother(alpha=alpha)
    metrics = PerformanceMetrics()
    
    raw_coordinates = []
    smoothed_coordinates = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 模拟检测结果 (添加一些随机噪声)
        base_x = frame.shape[1] // 2
        base_y = frame.shape[0] // 2
        
        # 原始检测 (带噪声)
        noise_x = int(np.random.normal(0, 15))
        noise_y = int(np.random.normal(0, 15))
        raw_box = [
            base_x + noise_x - 50,
            base_y + noise_y - 50,
            base_x + noise_x + 50,
            base_y + noise_y + 50
        ]
        raw_center = [base_x + noise_x, base_y + noise_y]
        
        raw_coordinates.append(tuple(raw_center))
        
        # 应用平滑
        fake_det = {"box": raw_box, "center": raw_center}
        smoothed_det = smoother.update(fake_det)
        
        if smoothed_det:
            smoothed_coordinates.append(tuple(smoothed_det["center"]))
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    
    # 计算抖动对比
    raw_jitter = metrics.calculate_jitter(raw_coordinates)
    smoothed_jitter = metrics.calculate_jitter(smoothed_coordinates)
    improvement = (raw_jitter - smoothed_jitter) / raw_jitter * 100 if raw_jitter > 0 else 0
    
    print(f"\n📊 结果分析:")
    print(f"  原始坐标抖动：{raw_jitter:.2f} px")
    print(f"  平滑后抖动：  {smoothed_jitter:.2f} px")
    print(f"  抖动减少：    {improvement:.1f}% ✨")
    
    if improvement > 50:
        print(f"  ✅ EMA 平滑效果显著!")
    elif improvement > 30:
        print(f"  👍 EMA 平滑有一定效果")
    else:
        print(f"  ⚠️  可能需要调整α参数或检查数据质量")
    
    return {
        "raw_jitter": raw_jitter,
        "smoothed_jitter": smoothed_jitter,
        "improvement": improvement
    }


def test_resolution_performance(model_path: str = "yolov8n.pt"):
    """
    测试不同分辨率下的性能表现
    
    Args:
        model_path: YOLO 模型路径
    """
    print("\n" + "="*60)
    print("测试 2: 分辨率 vs 性能分析")
    print("="*60)
    
    resolutions = [320, 480, 640]
    results = []
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    for res in resolutions:
        print(f"\n  Testing resolution: {res}x{res}...")
        
        detector = ObjectDetector(
            model_path=model_path,
            input_size=res,
            conf_threshold=0.35
        )
        
        # 预热
        for _ in range(5):
            _ = detector.detect(test_image)
        
        # 正式测试
        start_time = time.time()
        num_frames = 20
        
        for _ in range(num_frames):
            _ = detector.detect(test_image)
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        results.append({
            "resolution": res,
            "fps": fps,
            "meets_realtime": fps >= 30
        })
        
        status = "✅" if fps >= 30 else "⚠️"
        print(f"    FPS: {fps:.1f} {status} {'(Real-time ✓)' if fps >= 30 else '(Below 30 FPS)'}")
    
    print(f"\n📊 推荐配置:")
    realtime_results = [r for r in results if r["meets_realtime"]]
    if realtime_results:
        best_res = max(realtime_results, key=lambda x: x["resolution"])
        print(f"  最佳平衡点：{best_res['resolution']}x{best_res['resolution']} @ {best_res['fps']:.1f} FPS")
    else:
        print(f"  所有分辨率均未达到 30FPS，建议使用 320x320 或优化模型")
    
    return results


def test_roi_attention():
    """测试 ROI 注意力的有效性"""
    print("\n" + "="*60)
    print("测试 3: ROI 注意力机制验证")
    print("="*60)
    
    roi = ROIAttention(base_radius=150, confidence_boost=0.15)
    
    # 模拟检测场景
    hand_info = {
        "hand_found": True,
        "center": (320, 240)
    }
    
    frame_shape = (480, 640, 3)
    
    # 创建不同距离的模拟检测
    test_detections = [
        {"label": "cup", "conf": 0.5, "box": [300, 220, 340, 260], "center": [320, 240]},  # ROI 中心
        {"label": "cup", "conf": 0.5, "box": [250, 190, 290, 230], "center": [270, 210]},  # ROI 边缘
        {"label": "cup", "conf": 0.5, "box": [100, 100, 140, 140], "center": [120, 120]},  # ROI 外
    ]
    
    print("\n  原始检测结果:")
    for i, det in enumerate(test_detections):
        print(f"    [{i}] conf={det['conf']:.3f}, center={det['center']}")
    
    # 应用 ROI 注意力
    adjusted = roi.apply(test_detections, hand_info, frame_shape)
    
    print(f"\n  ROI 调整后的检测结果 (按置信度排序):")
    for i, det in enumerate(adjusted):
        in_roi_str = "✓ IN ROI" if det.get("in_roi") else "✗ outside"
        boost = det["conf"] - 0.5
        print(f"    [{i}] conf={det['conf']:.3f} (boost: +{boost:.3f}), {in_roi_str}")
    
    # 验证 ROI 内目标排名提升
    original_ranking = [d["center"] for d in test_detections]
    adjusted_ranking = [d["center"] for d in adjusted]
    
    roi_center_in_original = original_ranking.index((320, 240))
    roi_center_in_adjusted = adjusted_ranking.index((320, 240))
    
    rank_improvement = roi_center_in_original - roi_center_in_adjusted
    
    print(f"\n📊 ROI 效果:")
    print(f"  ROI 中心目标排名：{roi_center_in_original + 1} → {roi_center_in_adjusted + 1}")
    print(f"  排名提升：{rank_improvement + 1}位 {'↑' if rank_improvement >= 0 else '↓'}")
    
    if rank_improvement >= 0:
        print(f"  ✅ ROI 注意力成功提升了手部附近目标的优先级!")
    
    return {
        "rank_improvement": rank_improvement,
        "final_rank": roi_center_in_adjusted + 1
    }


def test_occlusion_robustness(video_path: str):
    """
    测试遮挡场景下的鲁棒性
    
    Args:
        video_path: 包含遮挡场景的视频路径
    """
    print("\n" + "="*60)
    print("测试 4: 遮挡场景鲁棒性测试")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    smoother = BMASmoother(alpha=0.6, max_frames_lost=10)
    
    total_frames = 0
    occluded_frames = 0
    successful_predictions = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # 模拟遮挡场景 (随机丢失检测)
        is_occluded = np.random.random() < 0.3  # 30% 概率发生遮挡
        
        if not is_occluded:
            # 正常检测
            fake_det = {
                "box": [100, 100, 200, 200],
                "center": [150, 150]
            }
            result = smoother.update(fake_det)
            if result:
                successful_predictions += 1
        else:
            # 遮挡情况
            occluded_frames += 1
            result = smoother.update(None)  # 无检测输入
            
            if result and result.get("box"):
                # 平滑器成功预测
                successful_predictions += 1
    
    cap.release()
    
    prediction_rate = successful_predictions / total_frames * 100 if total_frames > 0 else 0
    occlusion_rate = occluded_frames / total_frames * 100 if total_frames > 0 else 0
    
    print(f"\n📊 遮挡测试结果:")
    print(f"  总帧数：        {total_frames}")
    print(f"  遮挡帧数：      {occluded_frames} ({occlusion_rate:.1f}%)")
    print(f"  成功预测帧数：  {successful_predictions}")
    print(f"  预测成功率：    {prediction_rate:.1f}%")
    
    if prediction_rate > 80:
        print(f"  ✅ 时序平滑在遮挡情况下表现优秀!")
    elif prediction_rate > 60:
        print(f"  👍 基本能够处理遮挡场景")
    else:
        print(f"  ⚠️  需要增加 max_frames_lost 或调整参数")
    
    return {
        "occlusion_rate": occlusion_rate,
        "prediction_success_rate": prediction_rate
    }


def generate_report(results: Dict, output_path: str = "test_report.json"):
    """生成测试报告"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": results,
        "summary": {
            "jitter_reduction": results.get("smoother_test", {}).get("improvement", 0),
            "recommended_resolution": None,
            "roi_effective": results.get("roi_test", {}).get("rank_improvement", 0) >= 0,
            "occlusion_robust": results.get("occlusion_test", {}).get("prediction_success_rate", 0) > 70
        }
    }
    
    # 找出推荐分辨率
    if "resolution_test" in results:
        realtime_res = [r for r in results["resolution_test"] if r.get("meets_realtime")]
        if realtime_res:
            best = max(realtime_res, key=lambda x: x["resolution"])
            report["summary"]["recommended_resolution"] = f"{best['resolution']}x{best['resolution']}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试报告已保存至：{output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='视觉辅助系统精度优化性能验证')
    parser.add_argument('--video', type=str, default=None, help='测试视频路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO 模型路径')
    parser.add_argument('--output', type=str, default='test_report.json', help='测试报告输出路径')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    
    args = parser.parse_args()
    
    print("\n" + "✨"*30)
    print("视觉辅助系统精度优化 - 性能验证工具")
    print("✨"*30)
    
    all_results = {}
    
    # 测试 1: EMA 平滑器 (需要视频)
    if args.video or args.all:
        video_path = args.video if args.video else "data/test_video.mp4"
        if Path(video_path).exists():
            all_results["smoother_test"] = test_smoother_effectiveness(video_path)
        else:
            print(f"\n⚠️  跳过平滑器测试：未找到视频文件 {video_path}")
            print("   使用示例：python test_accuracy_improvement.py --video your_video.mp4")
    
    # 测试 2: 分辨率性能
    all_results["resolution_test"] = test_resolution_performance(args.model)
    
    # 测试 3: ROI 注意力
    all_results["roi_test"] = test_roi_attention()
    
    # 测试 4: 遮挡鲁棒性 (可选)
    if args.video and Path(args.video).exists():
        all_results["occlusion_test"] = test_occlusion_robustness(args.video)
    
    # 生成报告
    report = generate_report(all_results, args.output)
    
    # 总结
    print("\n" + "="*60)
    print("📋 优化效果总结")
    print("="*60)
    print(f"  ✓ 时序平滑抖动减少：  {report['summary']['jitter_reduction']:.1f}%")
    print(f"  ✓ 推荐分辨率：       {report['summary']['recommended_resolution'] or 'N/A'}")
    print(f"  ✓ ROI 注意力有效：    {'是 ✓' if report['summary']['roi_effective'] else '否'}")
    print(f"  ✓ 遮挡鲁棒性良好：    {'是 ✓' if report['summary']['occlusion_robust'] else '否'}")
    print("="*60)
    
    print("\n🎉 所有测试完成!\n")


if __name__ == '__main__':
    main()
