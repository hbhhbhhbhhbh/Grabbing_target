# 🚀 环境设置完整指南

## 📋 方式一：使用自动化脚本（推荐）

### 1. 运行自动安装脚本

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

脚本会自动：
- ✅ 创建 conda 环境 (vision_optimized)
- ✅ 安装所有依赖包
- ✅ 验证安装是否成功

### 2. 激活环境

```bash
conda activate vision_optimized
```

### 3. 运行验证测试

```bash
python test_environment.py
```

如果看到所有✅，说明环境设置成功！

---

## 📋 方式二：手动逐步安装

### 步骤 1: 创建 Conda 环境

```bash
conda create -n vision_optimized python=3.11 -y
```

### 步骤 2: 激活环境

```bash
conda activate vision_optimized
```

### 步骤 3: 安装核心依赖

```bash
pip install opencv-python ultralytics mediapipe numpy
```

**⏰ 预计时间**: 5-15 分钟（取决于网络速度）

**💡 提示**: 如果下载速度慢，可以使用国内镜像：

```bash
pip install opencv-python ultralytics mediapipe numpy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 步骤 4: 验证安装

```bash
python test_environment.py
```

---

## ✅ 成功的标志

运行 `test_environment.py` 后应该看到：

```
✅ OpenCV 导入成功
✅ NumPy 导入成功
✅ Ultralytics YOLO 导入成功
   ✅ 模型文件加载成功
✅ MediaPipe 导入成功
   ✅ 手部检测器创建成功

测试自定义模块...
✅ temporal_smoother 模块导入成功
   ✅ EMA 平滑器工作正常
✅ roi_attention 模块导入成功
   ✅ ROI 注意力机制工作正常
✅ object_detector 模块导入成功
   ✅ 置信度阈值已优化 (0.35)
   ✅ NMS IoU 阈值参数已添加
   ✅ 输入分辨率参数已添加
✅ hand_tracker 模块导入成功
   ✅ 手部检测阈值已优化 (0.4)

环境验证完成!
```

---

## 🎯 运行主程序

环境验证成功后，就可以运行优化后的系统了：

```bash
conda activate vision_optimized
python main_step9_state_machine.py
```

---

## 🔧 常见问题排查

### ❌ 问题 1: pip 安装超时或失败

**原因**: 网络连接问题或 PyPI 服务器响应慢

**解决方案**:
```bash
# 使用清华镜像源
pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
```

---

### ❌ 问题 2: 找不到 yolov8n.pt 模型文件

**解决方案**:
```bash
# 方法 1: 自动下载（首次运行时会自动下载）
# 确保当前目录有 yolov8n.pt

# 方法 2: 手动下载
cd /Users/lidongrui/Documents/comp5523/vision_project
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o yolov8n.pt
```

---

### ❌ 问题 3: MediaPipe 手部模型文件不存在

**解决方案**:
确认 `hand_landmarker.task` 文件在项目目录中：
```bash
ls -lh hand_landmarker.task
```

如果不存在，需要从项目备份中复制或重新下载。

---

### ❌ 问题 4: ImportError: No module named 'xxx'

**可能原因**: 
- 没有激活正确的 conda 环境
- 安装过程中断

**解决方案**:
```bash
# 1. 确认当前环境
conda info --envs

# 2. 激活正确的环境
conda activate vision_optimized

# 3. 重新安装
pip install -r requirements.txt
```

---

## 💾 环境信息

- **Python 版本**: 3.11.x
- **Conda 环境**: vision_optimized
- **核心依赖**:
  - opencv-python >= 4.8.0
  - ultralytics >= 8.0.0
  - mediapipe >= 0.10.0
  - numpy >= 1.24.0

---

## 📊 磁盘空间需求

| 组件 | 大小 |
|------|------|
| Conda 环境基础 | ~500 MB |
| PyTorch | ~2 GB |
| OpenCV | ~100 MB |
| MediaPipe | ~200 MB |
| YOLO 模型 | ~6 MB |
| **总计** | **~3 GB** |

请确保至少有 **5 GB** 的可用磁盘空间。

---

## 🎉 完成！

环境设置完成后，您就可以：

1. **测试优化效果**:
   ```bash
   python tests/test_accuracy_improvement.py --all
   ```

2. **运行主程序**:
   ```bash
   python main_step9_state_machine.py
   ```

3. **数据增强** (如需微调):
   ```bash
   python tools/augment_for_egocentric.py \
       --input data/original \
       --output data/augmented \
       --motion-blur --cutout --num-aug 10
   ```

---

## 📖 参考文档

- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 优化总结
- [docs/dataset_preparation.md](docs/dataset_preparation.md) - 数据集准备

---

*最后更新：2026-03-19*  
*适用系统：macOS (Apple Silicon)*
