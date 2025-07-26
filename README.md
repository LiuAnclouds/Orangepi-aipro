# Orangepi-aipro-ffmpeg-detect-yolo

## 声明

本项目用于OrangePi-AiPro的yolo推理部署，包括对视频流、图片、摄像头的推流，可以实现NPU加速推理，推流拉流的视线，开机自启动等服务程序，是一个基于香橙派AIPRO的目标检测项目。后续作者会上传对应的镜像，烧录后可直接使用。

* 只需要推流服务的只需看一
* 需要推流+自启动的需要看一和二
* 只需要YOLO+NPU加速的只需看三
* 需要YOLO+推流的只需看三
* 需要YOLO+推流+自启动的看二、三、四、五

项目结构如下:

```bash
│  LICENSE
│  README.md
│
├─ffmpeg
│  ├─scripts
│  │      command_only_push.sh
│  │      command_push_detect.sh
│  │      detect_om_camera.service
│  │      detect_om_camera.sh
│  │      ffmpeg_rtsp.sh
│  │      ffmpeg_stream.service
│  │      mediamtx.service
│  │
│  └─settings
│          addfile.tar.gz
│          ffmpeg_4.4.2-1_arm64.deb
│          mediamtx_v1.8.4_linux_arm64v8.tar.gz
│          testvideo.mp4
│
└─yolo-npu
        coco.yaml
        detect_om_camera.py
        detect_om_camera_pushstream.py
        detect_om_pic.py
        detect_om_video.py
        detect_pho.py
        pt2onnx.py
        yolov8n.pt
```



# 一、推流环境安装

## 1.安装ffmpeg并用Ascend加速

```bash
cd /home/HwHiAiUser
#克隆项目
git clone https://github.com/LiuAnclouds/Orangepi-aipro.git
cd Orangepi-aipro
# 更换目录
cd /ffmpeg/settings
cp *.* /home/HwHiAiUser
# 所有的安装都是基于HwHiAiUser目录
cd /home/HwHiAiUser
#配置bashrc
vim .bashrc
#将以下内容添加进bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
#重启bashrc
source .bashrc
#安装ffmpeg
sudo dpkg -i ffmpeg_4.4.2-1_arm64.deb
#验证安装
ffmpeg -hwaccels | grep ascend
#ascend
ffmpeg -encoders | grep ascend
#V..... h264_ascend Ascend HiMpi H264 encoder (codec h264)
#V..... h265_ascend Ascend HiMpi H265 encoder (codec hevc)
ffmpeg -decoders | grep ascend
#V..... h264_ascend Ascend HiMpi H264 decoder (codec h264)
#V..... h265_ascend Ascend HiMpi H265 decoder (codec hevc)
# 源码构建
# 取消deb-src注释
sudo vim /etc/apt/sources.list
#构建
sudo apt update
sudo apt-get install dpkg-dev
apt source ffmpeg
#检验
ls | grep ffmpeg
#补丁文件
tar zxf addfile.tar.gz
cp -r addfile/* ./ffmpeg-4.4.2
#将以下内容添加进bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
#重启bashrc
source .bashrc
#编译带有dvpp功能的ffmpeg包
cd ffmpeg-4.4.2
./configure --prefix=/usr --enable-shared --extra-cflags="-I${ASCEND_HOME_PATH}/acllib/include" --extra-ldflags="-L${ASCEND_HOME_PATH}/aarch64-linux/lib64" --extra-libs="-lacl_dvpp_mpi -lascendcl" --enable-ascend
make -j4
sudo make install
#验证安装
cd /home/HwHiAiUser
ffmpeg -hwaccels | grep ascend
#ascend
ffmpeg -encoders | grep ascend
#V..... h264_ascend Ascend HiMpi H264 encoder (codec h264)
#V..... h265_ascend Ascend HiMpi H265 encoder (codec hevc)
ffmpeg -decoders | grep ascend
#V..... h264_ascend Ascend HiMpi H264 decoder (codec h264)
#V..... h265_ascend Ascend HiMpi H265 decoder (codec hevc)
#应用测试
ffmpeg -i testvideo.mp4 -vcodec h264_ascend test.264
```

## 2.推流配置

```bash
#所有配置子目录依然在该目录下
cd /home/HwHiAiUser/
#解压
mkdir mediamtx
tar -zxf mediamtx_v1.8.4_linux_arm64v8.tar.gz -C mediamtx
cp mediamtx/mediamtx.yml ./
#查询IP
ifconfig
#启动服务
vim ./mediamtx.yml #特殊需求更改配置
sudo mediamtx/mediamtx #启动服务
#推流
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 -i /dev/video0 -c:v h264_ascend -b:v 1000k -maxrate 5000k -bufsize 256k -g 5 -keyint_min 5 -rtsp_transport udp -f rtsp rtsp://10.31.3.120:8554/mystream1  #ip内容自行更换
```

经过以上配置，有关推流的内容已经全部配置完成，如果还有自启动服务或者目标检测功能需求的小伙伴可以继续向下观看，如果无需求，可以停止，接下来将依次配置，推流自启动服务，目标检测+推流自启动服务。

# 二、自启动推流服务

```bash
├─scripts
│      command_only_push.sh
│      command_push_detect.sh
│      detect_om_camera.service
│      detect_om_camera.sh
│      ffmpeg_rtsp.sh
│      ffmpeg_stream.service
│      mediamtx.service
│
└─settings
        addfile.tar.gz
        ffmpeg_4.4.2-1_arm64.deb
        mediamtx_v1.8.4_linux_arm64v8.tar.gz
        testvideo.mp4
```

整个推流的搭建都依赖于以上目录，服务脚本和运行脚本都位于scripts，如有特殊需求自行修改

## 1.配置服务文件

### 1.修改内容

#### 1.mediamtx.service

```bash
[Unit]
Description=MediaMTX RTSP Server
After=network.target

[Service]
User=HwHiAiUser
WorkingDirectory=/home/HwHiAiUser/mediamtx
ExecStart=/home/HwHiAiUser/mediamtx/mediamtx
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

```

如果路径不一样需要将WorkingDirectory和ExecStart改为自己路径，默认情况不需要修改

#### 2.ffmpeg_stream.service

```bash
[Unit]
Description=FFmpeg RTSP Stream
After=mediamtx.service
Requires=mediamtx.service

[Service]
User=HwHiAiUser
Group=HwHiAiUser
WorkingDirectory=/home/HwHiAiUser
EnvironmentFile=/usr/local/Ascend/ascend-toolkit/set_env.sh
Environment="LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH"
Environment="PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/Ascend/ascend-toolkit/latest/bin"
ExecStart=/bin/bash /home/HwHiAiUser/ffmpeg_rtsp.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

```

这里需要修改ExecStart中的/home/HwHiAiUser/ffmpeg_rtsp.sh部分，如果需要自启动就运行的脚本不在此地方，需要自己修改路径。修改完以上内容推流自启动服务文件就算完成了。

```bash
# 将服务脚本转移到服务文件夹
cd /home/HiHwAiUser/Orangepi-aipro/ffmpeg/scrpts
cp mediamtx.service /etc/systemd/system
cp ffmpeg_stream.service /etc/systemd/system
# 增加权限
chmod 777 /etc/systemd/system/*.service
```

## 2.配置自启动脚本

现在需要配置我的自启动需要启动哪些内容，即ffmpeg_rtsp.sh

```bash
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
    -f rtsp "rtsp://192.168.1.1:8554/main.264" \
    >> /tmp/ffmpeg_rtsp.log 2>&1

```

这里需要修改 -f rtsp "rtsp://192.168.1.1:8554/main.264" 中的IP地址，修改为自己香橙派所需要的地址即可。修改完成之后，我们就可以启动服务了。切记不要忘记插上摄像头！

```bash
cp ffmpeg_rtsp.sh /home/HwHiAiUser
```



## 3.启动服务脚本

```bash
cp command_only_push.sh /home/HwHiAiUser
#更换目录
cd /home/HwHiAiUser
#这个命令是将Windows下的字符换为Unix格式，如果出现字符格式错误请用该命令重新生成
sed -i 's/\r$//' /home/HwHiAiUser/command_only_push.sh
#权限
chmod 777 command_only_push.sh
#运行启动命令脚本
./command_only_push.sh
```

这时候就可以查看状态了，如果是activate表明已经全部激活，如果出现问题可前往/temp查看日志，没有问题就可以看到已经推流成功，我们只需要在拉流软件上，输入我们的地址即可

```bash
rtsp://192.168.1.1:8554/main.264
```

如果只需要自启动服务+推流的小伙伴已经完成以上内容，接下来的内容可省略，下面内容为将目标检测的结果自启动并进行推流。

# 三、NPU加速YOLO环境配置

## 1.conda

```bash
# 复制base环境
conda create -n yolov8 --clone base
# 激活
conda activate yolov8
#配置清华源
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
#安装ultralytics
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple #使用清华源
#安装onnx
pip install onnx onnxruntime
```

## 2.pt2onnx

```bash
cd /home/HwHiAiUser/Orangepi-aipro/yolo-npu
```

修改pt2onnx.py的权重路径

```python
from ultralytics import YOLO


model = YOLO("yolov8n.pt")  

# Export the model
model.export(format="onnx",opset=12)
```

```bash
# 将pt转为onnx
python pt2onnx.py
```

## 3.测试onnx

```bash
#修改对应的detect_pho中的onnx路径，然后运行即可
python detect_pho.py
```

## 4.onnx2om

```bash
#运行该命令将onnx转为om
atc --model=yolov8n.onnx --framework=5 --output=yolov8n --input_format=NCHW --input_shape="images:1,3,640,640" --log=error --soc_version=Ascend310B4 --output_type=FP32
```

## 5.摄像头推理测试

* detect_om_camera是摄像头推理

* detect_om_pic是图片推理

* detect_om_video是视频推理

* detect_om_camera_pushstream是摄像头推理+推流

  更换模型权重的话对应的类别数据也要改变，即coco.yaml改为自己的类别即可

```bash
python detect_om_camera.py
python detect_om_camera_pushstream.py
```

此时就可以看到有npu加速的推理图像了，如果出现/x0a8字符问题，将开头的code-utf8删掉即可

# 四、YOLO推理自启动

## 1.配置服务文件

### 修改服务内容

mediamtx.service同二同理，这里只需要改detect_om_camera.service即可

```bash
[Unit]
Description=FFmpeg RTSP Stream
After=mediamtx.service
Requires=mediamtx.service

[Service]
User=HwHiAiUser
Group=HwHiAiUser
WorkingDirectory=/home/HwHiAiUser
EnvironmentFile=/usr/local/Ascend/ascend-toolkit/set_env.sh
Environment="LD_LIBRARY_PATH=/usr/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH"
Environment="PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/Ascend/ascend-toolkit/latest/bin"
ExecStart=/bin/bash /home/HwHiAiUser/detect_om_camera.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

```

这里依然是只需要修改/home/HwHiAiUser/detect_om_camera.sh，如果脚本路径变换，只需修改路径即可

```bash
cd /home/HiHwAiUser/Orangepi-aipro/ffmpeg/scrpts
cp mediamtx.service /etc/systemd/system
cp detect_om_camera.service /etc/systemd/system
# 增加权限
chmod 777 /etc/systemd/system/*.service
```

## 2.配置自启动脚本

这时候需要修改detecm_om_camera.sh中的内容

```bash
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
source /usr/local/miniconda3/etc/profile.d/conda.sh #这里更换conda路径

# 激活yolov8-2环境
echo "Activating conda environment yolov8..."
conda activate yolov8	#这里更换环境名称
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment yolov8-2"
    exit 1
fi

# 验证Python路径
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# 切换到工作目录
cd /home/HwHiAiUser/Orangepi-aipro/yolo-npu #这里更换工作目录
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to cd /home/HwHiAiUser/Orangepi-aipro/yolo-npu directory"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "detect_om_camera.py" ]; then
    echo "ERROR: detect_om_camera.py not found in $(pwd)"
    exit 1
fi

# 运行Python脚本
echo "Starting Python script..."
python detect_om_camera_pushstream.py #这里更换自启动的程序
```

需要更改的地方已经在内容里标注，这里简单说明一下：

* 1.conda环境路径
* 2.conda环境名称
* 3.工作目录
* 4.自启动程序

```bash
cd /home/HiHwAiUser/Orangepi-aipro/ffmpeg/scrpts
cp detect_om_camera.sh /home/HwHiAiUser
chmod 777 detect_om_camera.sh
```



## 3.启动服务脚本

```bash

cp command_push_detect.sh /home/HwHiAiUser
#更换目录
cd /home/HwHiAiUser
#这个命令是将Windows下的字符换为Unix格式，如果出现字符格式错误请用该命令重新生成
sed -i 's/\r$//' /home/HwHiAiUser/command_push_detect.sh
#权限
chmod 777 command_push_detect.sh
#运行启动命令脚本
./command_push_detect.sh
```

这时候就可以查看状态了，如果是activate表明已经全部激活，如果出现问题可前往/temp查看日志，没有问题就可以看到已经推流成功，我们只需要在拉流软件上，输入我们的地址即可

```bash
rtsp://192.168.1.1:8554/main.264
```

到此就全部结束了，中间路径问题有一些繁琐，大家可以自行修改，只要成功链接就可以。