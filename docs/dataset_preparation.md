# 第一视角数据集准备指南

本指南介绍如何为视障辅助抓取系统准备和标注第一视角 (Egocentric) 训练数据。

## 为什么需要第一视角数据？

基础的预训练模型（如在 COCO 数据集上训练的 YOLO）通常基于**第三人称平视视角**,在视障人士的第一视角场景中会出现严重的水土不服:

- **视角差异**: 俯视角度导致物体形变
- **相机抖动**: 可穿戴设备或手机必然产生拖影
- **手部遮挡**: 抓取动作最后阶段，手会遮挡 50% 以上的物体面积
- **距离变化**: 物体从远到近快速缩放

## 推荐数据集

### 1. EPIC-Kitchens

**简介**: 大规模第一视角厨房活动数据集，包含日常物品的高质量标注

**数据特点**:
- 55 小时第一视角视频
- 100+ 个日常物体类别 (杯子、瓶子、餐具等)
- 自然抓取和操作动作
- 分辨率：1920x1080

**下载链接**:
- 官网：https://epic-kitchens.github.io/
- Google Drive / Baidu Netdisk (需申请)

**使用许可**: CC-BY 4.0

---

### 2. Ego4D

**简介**: 超大规模第一视角日常生活数据集

**数据特点**:
- 3670 小时第一视角视频
- 涵盖多种日常场景
- 包含手部 - 物体交互标注
- 分辨率：多样化

**下载链接**:
- 官网：https://ego4d-data.org/
- 需要学术申请

**使用许可**: 研究用途

---

### 3. 自建数据集 (推荐)

**优势**: 
- 针对具体应用场景定制
- 可以控制光照、物体种类、背景复杂度
- 更符合视障用户的实际使用环境

**采集建议**:

#### 设备选择
| 设备类型 | 优点 | 缺点 | 推荐度 |
|---------|------|------|--------|
| GoPro/运动相机 | 轻便、广角、防抖好 | 需要额外固定 | ⭐⭐⭐⭐⭐ |
| 手机前置摄像头 | 易获取、质量高 | 重量较大 | ⭐⭐⭐⭐ |
| RealSense D435i | 深度信息、RGB-D | 成本高 | ⭐⭐⭐ |

#### 采集场景清单
- [ ] 厨房桌面 (杯子、碗、餐具)
- [ ] 书桌 (手机、钥匙、笔)
- [ ] 浴室 (洗漱用品)
- [ ] 客厅遥控器、水杯等

#### 采集动作设计
1. **搜索阶段**: 相机缓慢平移扫描桌面
2. **接近阶段**: 手伸向目标物体
3. **抓取阶段**: 手指张开→闭合→握紧
4. **移动阶段**: 抓起物体后移动

**每场景建议数量**:
- 静态场景：50-100 张
- 动态视频：10-20 段 (每段 5-10 秒)
- 总帧数：至少 2000 帧有效样本

---

## 数据标注格式转换

### YOLO 格式标注规范

```
# 每张图像对应一个 .txt 文件
# 文件名：image_001.jpg → image_001.txt
# 每行一个目标：<class_id> <x_center> <y_center> <width> <height>

# 示例 (假设图片尺寸 640x480):
# 一个杯子位于图像中央，宽 100px，高 150px
0 0.5 0.5 0.15625 0.3125

# 说明:
# - class_id: 类别编号 (从 0 开始)
# - x_center, y_center: 归一化中心坐标 (0-1)
# - width, height: 归一化宽高 (0-1)
```

### 类别映射建议

针对视障辅助抓取场景，推荐以下核心类别:

| ID | 类别名称 | COCO 映射 | 说明 |
|----|---------|----------|------|
| 0 | cup | cup (41) | 各种水杯、马克杯 |
| 1 | bottle | bottle (39) | 饮料瓶、调味瓶 |
| 2 | cell phone | cell phone (67) | 手机 |
| 3 | keys | - | 钥匙串 (COCO 无，需自定义) |
| 4 | spoon | spoon (63) | 勺子 |
| 5 | fork | fork (61) | 叉子 |
| 6 | knife | knife (62) | 刀 |
| 7 | bowl | bowl (42) | 碗 |
| 8 | remote | remote (74) | 遥控器 |

### 标注工具推荐

#### 1. LabelImg (轻量级)
```bash
pip install labelimg
labelimg
```
- 支持 YOLO 格式导出
- 适合静态图像标注
- GitHub: https://github.com/tzutalin/labelImg

#### 2. CVAT (专业级)
- 在线标注平台
- 支持视频标注和插值
- 团队协作功能
- 网址：https://cvat.ai/

#### 3. Roboflow (一站式)
- 数据管理 + 标注 + 增强
- 直接导出 YOLO 格式
- 免费版支持公开项目
- 网址：https://roboflow.com/

---

## 数据增强策略

### 使用本项目提供的增强工具

```bash
# 基本用法 (启用所有增强)
python tools/augment_for_egocentric.py \
    --input data/original \
    --output data/augmented \
    --num-aug 5

# 仅启用关键增强 (运动模糊 + 遮挡)
python tools/augment_for_egocentric.py \
    --input data/original \
    --output data/augmented \
    --motion-blur \
    --cutout \
    --num-aug 10

# 自定义参数
python tools/augment_for_egocentric.py \
    --input data/original \
    --output data/augmented \
    --motion-blur --cutout --noise --brightness \
    --num-aug 8 \
    --no-annotations
```

### 增强配置说明

```python
aug_config = {
    # 运动模糊 (模拟相机抖动)
    'motion_blur': True,
    'motion_blur_prob': 0.6,      # 60% 概率应用
    'max_kernel_size': 15,        # 最大模糊核尺寸
    
    # Cutout 遮挡 (强迫学习局部特征)
    'cutout': True,
    'cutout_prob': 0.7,           # 70% 概率应用
    'n_holes': 3,                 # 3 个随机遮挡区域
    'max_h_size': 50,             # 最大遮挡块 50x50px
    
    # 高斯噪声 (传感器噪声模拟)
    'noise': True,
    'noise_prob': 0.3,
    'noise_sigma': 25,
    
    # 亮度调整 (光照变化)
    'brightness': True,
    'brightness_prob': 0.5,
    
    # 随机旋转 (小角度)
    'rotation': True,
    'rotation_prob': 0.3,
    'max_rotation_angle': 30
}
```

### 增强倍数建议

| 原始数据量 | 建议增强倍数 | 最终数据量 | 适用场景 |
|-----------|------------|-----------|---------|
| 100 张 | 10x | 1100 张 | 快速原型验证 |
| 500 张 | 5x | 3000 张 | 课程项目 |
| 2000 张 | 3x | 8000 张 | 毕业设计 |
| 5000+ 张 | 2x | 15000+ 张 | 科研论文 |

---

## YOLO 模型微调流程

### 1. 准备数据集结构

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

### 2. 编写 data.yaml

```yaml
# dataset/data.yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

# 类别定义
names:
  0: cup
  1: bottle
  2: cell phone
  3: keys
  4: spoon
  5: fork
  6: knife
  7: bowl
  8: remote

nc: 9  # 类别数量
```

### 3. 开始微调训练

```bash
# 使用 YOLOv8n.pt 预训练权重
yolo detect train \
    model=yolov8n.pt \
    data=dataset/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=8 \
    optimizer=SGD \
    lr0=0.01 \
    patience=50 \
    save_period=10

# 关键参数说明:
# - epochs: 训练轮数 (根据数据量调整)
# - imgsz: 输入分辨率 (640 平衡速度与精度)
# - batch: 批次大小 (根据显存调整)
# - lr0: 初始学习率
# - patience: 早停耐心值
```

### 4. 训练监控

```bash
# 查看训练日志
tensorboard --logdir runs/detect/train

# 关键指标:
# - mAP@50: IoU=0.5 时的平均精度
# - mAP@50-95: 不同 IoU 阈值下的平均精度
# - precision: 查准率
# - recall: 查全率 (视障场景更重要)
# - loss: 损失函数收敛情况
```

### 5. 模型评估与导出

```bash
# 在验证集上评估
yolo detect val \
    model=runs/detect/train/weights/best.pt \
    data=dataset/data.yaml

# 导出为 ONNX 格式 (可选，用于部署优化)
yolo export \
    model=runs/detect/train/weights/best.pt \
    format=onnx \
    imgsz=640
```

---

## 数据质量检查清单

在开始训练前，请确保:

- [ ] 所有图像都有对应的标注文件
- [ ] 标注文件中的坐标已归一化 (0-1)
- [ ] 类别 ID 与 data.yaml 中的定义一致
- [ ] 训练集和验证集没有数据泄露
- [ ] 每个类别的样本数量相对均衡
- [ ] 标注框准确覆盖目标物体 (IoU > 0.8)
- [ ] 处理了极端情况 (严重遮挡、模糊、暗光)

---

## 常见问题解答

### Q1: 数据量太少怎么办？
**A**: 
1. 优先使用预训练模型 (COCO weights)
2. 应用激进的数据增强 (10x+)
3. 冻结骨干网络，只训练检测头
4. 使用迁移学习策略

### Q2: 小目标检测效果差？
**A**:
1. 提高输入分辨率 (640 → 1280)
2. 增加小目标样本比例
3. 使用 FPN/PANet 等多尺度特征融合
4. 调整 anchor boxes 尺寸

### Q3: 遮挡严重导致漏检？
**A**:
1. 强化 Cutout 数据增强
2. 降低推理时的 conf_threshold (0.5 → 0.35)
3. 使用时序平滑 (EMA) 利用历史帧信息
4. 考虑引入注意力机制

### Q4: 训练过拟合怎么办？
**A**:
1. 增加数据增强强度
2. 添加 Dropout/DropPath
3. 提前停止 (Early Stopping)
4. L2 正则化 (Weight Decay)

---

## 参考资源

### 论文
- EPIC-Kitchens: https://openaccess.thecvf.com/content_ECCV_2018/html/Dima_Damen_Scaling_Egocentric_Vision_ECCV_2018_paper.html
- Ego4D: https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html

### 代码库
- YOLOv8 官方：https://github.com/ultralytics/ultralytics
- MMDetection: https://github.com/open-mmlab/mmdetection

### 数据集下载
- EPIC-Kitchens: https://epic-kitchens.github.io/
- Ego4D: https://ego4d-data.org/
- HOI4D: https://hoi4d.github.io/ (第一视角人手 - 物体交互)

---

## 联系与支持

如有问题，请参考:
- Ultralytics YOLO 文档：https://docs.ultralytics.com/
- CVPR/ICCV/ECCV 最新第一视角视觉论文

**祝数据准备顺利！🎉**
