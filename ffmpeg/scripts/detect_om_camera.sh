#!/bin/bash

# 设置调试模式和日志
set -x
exec > >(tee -a /tmp/detect_om_camera.log) 2>&1
echo "=== Starting detect_om_camera service at $(date) ==="

# 设置Ascend环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/lib
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH

# 使用正确的conda路径
source /usr/local/miniconda3/etc/profile.d/conda.sh

# 激活yolov8-2环境
echo "Activating conda environment yolov8-2..."
conda activate yolov8-2
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment yolov8-2"
    exit 1
fi

# 验证Python路径
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# 切换到工作目录
cd /home/HwHiAiUser
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to /home/HwHiAiUser directory"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "detect_om_camera.py" ]; then
    echo "ERROR: detect_om_camera.py not found in $(pwd)"
    exit 1
fi

# 运行Python脚本
echo "Starting Python script..."
python detect_om_camera.py