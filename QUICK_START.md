# 🚀 快速启动指南 - 视觉辅助系统精度优化版

## ⚡ 5 分钟快速开始

### 1️⃣ 运行优化后的系统

```bash
python main_step9_state_machine.py
```

**你会看到**:
- 实时视频流显示
- 青色 ROI 圆圈 (检测到手时)
- 黄色检测框 (ROI 内目标)
- 绿色检测框 (ROI 外目标)
- 紫色框 + 平滑坐标 (选定目标)
- 右上角 FPS 计数器

---

### 2️⃣ 调整参数 (可选)

#### 提高速度 (如果 FPS < 30)

编辑 `main_step9_state_machine.py` 第 14-21 行:

```python
detector = ObjectDetector(
    model_path="yolov8n.pt",
    target_labels=["cup"],
    conf_threshold=0.35,
    iou_threshold=0.45,
    input_size=480,          # 👈 从 640 改为 480 (提速)
    label_weights={"cup": 1.2}
)
```

#### 提升精度 (如果硬件足够)

```python
input_size=640,             # 👈 保持 640 高分辨率
conf_threshold=0.30,        # 👈 进一步降低阈值 (更敏感)
```

#### 调整平滑强度

```python
box_smoother = BMASmoother(
    alpha=0.6,              # 👈 范围 0-1，越大越跟随当前帧
    max_frames_lost=5       # 👈 检测丢失后外推的最大帧数
)
```

---

### 3️⃣ 性能测试

```bash
# 运行完整测试套件
python tests/test_accuracy_improvement.py --all

# 查看单项测试
python tests/test_accuracy_improvement.py --video your_video.mp4
```

**测试输出示例**:
```
✨ 测试 1: EMA 时序平滑效果评估
  原始坐标抖动：12.34 px
  平滑后抖动：4.56 px
  抖动减少：63.1% ✨
  ✅ EMA 平滑效果显著!

✨ 测试 2: 分辨率 vs 性能分析
  320x320: 85.2 FPS ✅
  480x480: 52.1 FPS ✅
  640x640: 31.8 FPS ✅
  
  推荐配置：640x640 @ 31.8 FPS
```

---

### 4️⃣ 数据增强 (为微调准备)

```bash
# 准备原始图像
mkdir -p data/original
# 将你的图像放入 data/original/

# 应用增强
python tools/augment_for_egocentric.py \
    --input data/original \
    --output data/augmented \
    --motion-blur --cutout \
    --num-aug 10
```

**生成的文件**:
- `data/augmented/img_001.jpg` (原始图)
- `data/augmented/img_001_aug00.jpg` (增强版 1)
- `data/augmented/img_001_aug01.jpg` (增强版 2)
- ...
- `data/augmented/augmentation_annotations.json` (标注信息)

---

## 🎯 关键参数速查表

### 检测器参数

| 参数 | 推荐值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `conf_threshold` | 0.35 | 置信度阈值 | 漏检多→降低 (0.30), 误检多→提高 (0.45) |
| `iou_threshold` | 0.45 | NMS IoU 阈值 | 重叠框多→降低 (0.35), 框太少→提高 (0.55) |
| `input_size` | 640 | 输入分辨率 | FPS 低→降低 (480), 精度差→提高 (768) |
| `alpha` | 0.6 | EMA 平滑系数 | 抖动大→降低 (0.4), 延迟高→提高 (0.8) |

### ROI 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `base_radius` | 150 | ROI 基础半径 (像素) |
| `min_radius` | 80 | 最小 ROI 半径 |
| `confidence_boost` | 0.15 | ROI 内置信度提升值 |

### 手部检测参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `min_hand_detection_confidence` | 0.4 | 手部检测阈值 |
| `min_hand_presence_confidence` | 0.4 | 手部存在阈值 |
| `min_tracking_confidence` | 0.4 | 手部跟踪阈值 |

---

## 🔧 常见问题排查

### ❌ FPS 太低 (< 25)

**解决方案**:
1. 降低分辨率：`input_size=480`
2. 使用更小的模型：`yolov8n.pt` → `yolov8n.pt` (已是最小)
3. 减少目标类别：`target_labels=["cup"]` (不要检测太多类)

---

### ❌ 检测框跳动严重

**解决方案**:
1. 降低 EMA 系数：`alpha=0.4` (更平滑)
2. 增加 `max_frames_lost=10` (更好的外推)
3. 检查是否 FPS 过低导致卡顿

---

### ❌ 经常漏检

**解决方案**:
1. 降低置信度阈值：`conf_threshold=0.30`
2. 增加类别权重：`label_weights={"cup": 1.5}`
3. 提高分辨率：`input_size=768`

---

### ❌ 手部检测不稳定

**解决方案**:
1. 降低手部阈值：`min_hand_detection_confidence=0.35`
2. 确保光照充足
3. 避免手部过度遮挡

---

### ❌ ROI 区域不合理

**解决方案**:
1. 调整基础半径：`base_radius=200` (更大范围)
2. 关闭自动衰减：修改 `roi_decay_rate=1.0` (保持固定大小)
3. 根据状态调整：使用 `set_state_dependent_params()` 方法

---

## 📊 性能基准参考

| 配置 | FPS | 适用场景 | 推荐度 |
|------|-----|----------|--------|
| 320x320 + YOLOv8n | 80+ | 超高速场景 | ⭐⭐⭐ |
| 480x480 + YOLOv8n | 50+ | 平衡速度与精度 | ⭐⭐⭐⭐⭐ |
| 640x640 + YOLOv8n | 30+ | 标准配置 | ⭐⭐⭐⭐ |
| 768x768 + YOLOv8n | 20- | 高精度静态场景 | ⭐⭐ |

*测试环境：MacBook Air M1, 16GB RAM*

---

## 🎓 进阶使用

### 自定义类别权重

```python
detector = ObjectDetector(
    model_path="yolov8n.pt",
    target_labels=["cup", "bottle", "cell phone"],
    label_weights={
        "cup": 1.5,        # 杯子权重最高 (最常用)
        "bottle": 1.2,     # 瓶子次之
        "cell phone": 1.0  # 手机正常权重
    }
)
```

### 多尺度 ROI 策略

```python
from perception.roi_attention import MultiScaleROIAttention

roi_attention = MultiScaleROIAttention(
    radii=[80, 150, 250],   # 三层 ROI 半径
)
```

### 动态调整参数

```python
# 根据状态机状态调整 ROI
if state == "SEARCHING":
    roi_attention.set_state_dependent_params("SEARCHING")
elif state == "GRABBING":
    roi_attention.set_state_dependent_params("GRABBING")
```

---

## 📁 完整文件清单

✅ **新增核心模块** (2 个):
- `perception/temporal_smoother.py` - EMA 时序平滑器
- `perception/roi_attention.py` - 手部 ROI 注意力

✅ **新增工具** (1 个):
- `tools/augment_for_egocentric.py` - 数据增强工具

✅ **新增文档** (2 个):
- `docs/dataset_preparation.md` - 数据集准备指南
- `OPTIMIZATION_SUMMARY.md` - 优化总结报告

✅ **新增测试** (1 个):
- `tests/test_accuracy_improvement.py` - 性能验证脚本

✅ **修改文件** (2 个):
- `perception/object_detector.py` - 推理参数优化
- `perception/hand_tracker.py` - 检测阈值调优
- `main_step9_state_machine.py` - 集成所有优化

---

## 🎉 开始你的优化之旅

现在你已经准备好了！按照以下步骤开始:

1. **运行系统**: `python main_step9_state_machine.py`
2. **观察效果**: 注意 ROI 可视化和平滑的坐标输出
3. **调整参数**: 根据你的硬件和使用场景微调
4. **准备数据**: 如需进一步提升，收集第一视角图像并微调
5. **运行测试**: `python tests/test_accuracy_improvement.py --all`

**祝你使用顺利！有任何问题请参考 OPTIMIZATION_SUMMARY.md 或查阅相关文档。** 🚀

---

*最后更新：2026-03-19*  
*版本：v2.0 (精度优化版)*
