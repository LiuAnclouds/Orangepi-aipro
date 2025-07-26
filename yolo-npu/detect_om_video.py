# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from time import time
from ais_bench.infer.interface import InferSession
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

# 类别名称从yaml文件加载
CLASSES = yaml_load(check_yaml('fire.yaml'))['names']

# 为每个类别分配一个颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 初始化模型
model = InferSession(device_id=0, model_path="best_1_inoutfp32.om")


# 绘制边框的函数
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# 推理主函数
def main(original_image):
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    begin_time = time()
    outputs = model.infer(feeds=blob, mode="static")
    end_time = time()
    print("om infer time:", end_time - begin_time)

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))


# 加载视频文件
video_path = './test_images/test.mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    raise IOError("Cannot open video file")

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {frame_width}, Frame height: {frame_height}")

# 设置视频输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))  # 输出视频文件

# 逐帧推理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 对当前帧进行推理
    main(frame)

    # 写入输出视频
    out.write(frame)

    # 显示处理后的帧
    cv2.imshow('Inference Video', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
