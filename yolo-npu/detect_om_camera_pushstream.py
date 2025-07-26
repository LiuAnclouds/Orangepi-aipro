import os
import subprocess
import threading

# Verify the path
print(os.environ['LD_LIBRARY_PATH'])
import cv2
import numpy as np
from time import time
from ais_bench.infer.interface import InferSession
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
model = InferSession(device_id=0, model_path="yolov8n.om")

class RTSPStreamer:
    def __init__(self, rtsp_url="rtsp://192.168.31.239:8554/main.264", width=1920, height=1080, fps=30):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        
    def start_stream(self):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'h264_ascend',
            '-b:v', '2000k',
            '-maxrate', '3000k', 
            '-bufsize', '4000k',
            '-g', '15',
            '-keyint_min', '15',
            '-rtsp_transport', 'udp',
            '-f', 'rtsp',
            self.rtsp_url
        ]
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        
    def write_frame(self, frame):
        if self.process and self.process.stdin:
            frame_resized = cv2.resize(frame, (self.width, self.height))
            self.process.stdin.write(frame_resized.tobytes())
            
    def stop_stream(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(original_image):
    print(original_image.shape[:2])
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    outputs = model.infer(feeds=blob, mode="static")

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.15:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.15, 0.45, 0.25)

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
    return original_image

# Initialize camera and streamer
camera = cv2.VideoCapture(0)
streamer = RTSPStreamer()

if camera.isOpened():
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    streamer.start_stream()
else:
    raise IOError("Cannot open the webcam")

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame width: {frame_width}, Frame height: {frame_height}")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            raise IOError("Cannot capture frame")
        
        begin = time()
        frame = main(frame)
        end = time()
        fps = 1 / (end - begin)
        
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stream the processed frame
        streamer.write_frame(frame)
        
        cv2.imshow("detect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    streamer.stop_stream()
    cv2.destroyAllWindows()
    print("Camera and Stream Released")