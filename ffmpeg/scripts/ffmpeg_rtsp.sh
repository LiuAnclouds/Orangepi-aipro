#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 明确设置LD_LIBRARY_PATH（移除$LD_LIBRARY_PATH引用）
export LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/lib
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH

# 调试信息
echo "=== ENVIRONMENT ===" > /tmp/ffmpeg_debug.log
env >> /tmp/ffmpeg_debug.log
echo "=== FFMPEG CHECK ===" >> /tmp/ffmpeg_debug.log
/usr/bin/ffmpeg -encoders 2>> /tmp/ffmpeg_debug.log

# 运行命令（先简化测试）
echo "Starting RTSP stream at $(date)" >> /tmp/ffmpeg_rtsp.log
/usr/bin/ffmpeg \
    -f v4l2 -input_format mjpeg \
    -video_size 1920x1080 -framerate 30 \
    -i /dev/video0 \
    -c:v h264_ascend \
    -b:v 2000k -maxrate 3000k -bufsize 4000k \
    -g 15 -keyint_min 15 \
    -rtsp_transport udp \
    -f rtsp "rtsp://192.168.144.30:8554/main.264" \
    >> /tmp/ffmpeg_rtsp.log 2>&1
