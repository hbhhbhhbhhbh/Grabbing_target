"""
环境验证脚本 - 测试所有模块能否正常导入和运行
"""
import sys
import os
print(f"Python 版本：{sys.version}")
print("="*60)

# 测试基础库
try:
    import cv2
    print("✅ OpenCV 导入成功")
    print(f"   版本：{cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV 导入失败：{e}")

try:
    import numpy as np
    print("✅ NumPy 导入成功")
    print(f"   版本：{np.__version__}")
except Exception as e:
    print(f"❌ NumPy 导入失败：{e}")

# 测试 YOLO
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO 导入成功")
    
    # 尝试加载模型
    try:
        model = YOLO("yolov8n.pt")
        print("   ✅ 模型文件加载成功")
    except Exception as e:
        print(f"   ⚠️  模型文件加载失败 (可能不存在): {e}")
        
except Exception as e:
    print(f"❌ Ultralytics 导入失败：{e}")

# 测试 MediaPipe
try:
    import mediapipe as mp
    print("✅ MediaPipe 导入成功")
    
    # 尝试创建手部检测器
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        if os.path.exists("hand_landmarker.task"):
            base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
            options = vision.HandLandmarkerOptions(base_options=base_options)
            detector = vision.HandLandmarker.create_from_options(options)
            print("   ✅ 手部检测器创建成功")
        else:
            print("   ⚠️  hand_landmarker.task 文件不存在")
    except Exception as e:
        print(f"   ⚠️  手部检测器创建失败：{e}")
        
except Exception as e:
    print(f"❌ MediaPipe 导入失败：{e}")

# 测试自定义模块
print("\n" + "="*60)
print("测试自定义模块...")

try:
    from perception.temporal_smoother import BMASmoother
    print("✅ temporal_smoother 模块导入成功")
    
    # 测试 EMA 平滑器
    smoother = BMASmoother(alpha=0.6)
    test_det = {"box": [100, 100, 200, 200], "center": [150, 150]}
    result = smoother.update(test_det)
    if result:
        print("   ✅ EMA 平滑器工作正常")
    else:
        print("   ⚠️  EMA 平滑器返回 None")
        
except Exception as e:
    print(f"❌ temporal_smoother 导入或测试失败：{e}")

try:
    from perception.roi_attention import ROIAttention
    print("✅ roi_attention 模块导入成功")
    
    # 测试 ROI 注意力
    roi = ROIAttention(base_radius=150)
    test_dets = [
        {"label": "cup", "conf": 0.5, "box": [300, 220, 340, 260], "center": [320, 240]}
    ]
    hand_info = {"hand_found": True, "center": (320, 240)}
    result = roi.apply(test_dets, hand_info, (480, 640, 3))
    if result and len(result) > 0:
        print("   ✅ ROI 注意力机制工作正常")
    else:
        print("   ⚠️  ROI 注意力测试结果异常")
        
except Exception as e:
    print(f"❌ roi_attention 导入或测试失败：{e}")

try:
    from perception.object_detector import ObjectDetector
    print("✅ object_detector 模块导入成功")
    
    # 检查参数是否已优化
    import inspect
    sig = inspect.signature(ObjectDetector.__init__)
    params = sig.parameters
    
    if 'conf_threshold' in params and params['conf_threshold'].default == 0.35:
        print("   ✅ 置信度阈值已优化 (0.35)")
    else:
        print("   ⚠️  置信度阈值可能未优化")
        
    if 'iou_threshold' in params:
        print("   ✅ NMS IoU 阈值参数已添加")
    else:
        print("   ⚠️  NMS IoU 阈值参数缺失")
        
    if 'input_size' in params:
        print("   ✅ 输入分辨率参数已添加")
    else:
        print("   ⚠️  输入分辨率参数缺失")
        
except Exception as e:
    print(f"❌ object_detector 导入或测试失败：{e}")

try:
    from perception.hand_tracker import HandTracker
    print("✅ hand_tracker 模块导入成功")
    
    # 检查参数是否已优化
    import inspect
    sig = inspect.signature(HandTracker.__init__)
    params = sig.parameters
    
    if 'min_hand_detection_confidence' in params and params['min_hand_detection_confidence'].default == 0.4:
        print("   ✅ 手部检测阈值已优化 (0.4)")
    else:
        print("   ⚠️  手部检测阈值可能未优化")
        
except Exception as e:
    print(f"❌ hand_tracker 导入或测试失败：{e}")

print("\n" + "="*60)
print("环境验证完成!")
print("="*60)

# 总结
print("\n📋 快速开始:")
print("  conda activate vision_optimized")
print("  python main_step9_state_machine.py")
print("\n💡 提示：如果看到模型文件不存在的警告，请确保:")
print("  1. yolov8n.pt 在当前目录")
print("  2. hand_landmarker.task 在当前目录")
print("")
