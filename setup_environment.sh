#!/bin/bash
# 视觉辅助系统精度优化版 - 环境设置脚本

echo "🚀 开始设置 conda 环境..."

# 1. 创建环境 (如果不存在)
if conda info --envs | grep -q vision_optimized; then
    echo "✅ Conda 环境 'vision_optimized' 已存在"
else
    echo "⏳ 创建 conda 环境 'vision_optimized' (Python 3.11)..."
    conda create -n vision_optimized python=3.11 -y
fi

# 2. 激活环境
echo "⏳ 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vision_optimized

# 3. 安装依赖
echo "⏳ 安装依赖包 (这可能需要几分钟)..."
pip install opencv-python ultralytics mediapipe numpy

# 4. 验证安装
echo ""
echo "📋 验证安装..."
python -c "import cv2; import ultralytics; import mediapipe; print('✅ 所有依赖安装成功!')"

echo ""
echo "🎉 环境设置完成!"
echo ""
echo "使用方法:"
echo "  conda activate vision_optimized"
echo "  python main_step9_state_machine.py"
echo ""
