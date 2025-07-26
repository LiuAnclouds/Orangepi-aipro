# -*- coding: utf-8 -*-
import cv2
import numpy as np
import onnxruntime as ort
import time  # 新增时间模块

class YOLOv8ONNX:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入尺寸
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_height, self.model_width = self.input_shape[2:]
        
        # COCO类别标签
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image):
        # 调整大小并保持宽高比
        h, w = image.shape[:2]
        scale = min(self.model_height / h, self.model_width / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.model_height, self.model_width, 3), 114, dtype=np.uint8)
        canvas[(self.model_height-new_h)//2:(self.model_height-new_h)//2 + new_h,
               (self.model_width-new_w)//2:(self.model_width-new_w)//2 + new_w] = resized
        
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, (scale, (self.model_height-new_h)//2, (self.model_width-new_w)//2)
    
    def postprocess(self, outputs, letterbox_info):
        scale, pad_top, pad_left = letterbox_info
        predictions = np.squeeze(outputs[0]).T
        
        scores = np.max(predictions[:, 4:], axis=1)
        keep = scores > self.conf_thres
        predictions = predictions[keep]
        scores = scores[keep]
        
        if len(predictions) == 0:
            return []
        
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  self.conf_thres, self.iou_thres)
        
        detections = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            cls_id = class_ids[i]
            detections.append([*box, score, cls_id])
            
        return detections
    
    def detect(self, image):
        blob, letterbox_info = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, letterbox_info)
    
    def draw_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = map(float, det)
            color = (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.class_names[int(cls_id)]}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

if __name__ == "__main__":
    detector = YOLOv8ONNX("yolov8n.onnx", conf_thres=0.5, iou_thres=0.5)
    cap = cv2.VideoCapture(0)
    
    # FPS计算变量
    prev_time = 0
    curr_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 开始计时
        start_time = time.time()
        
        # 执行检测
        detections = detector.detect(frame)
        result_frame = detector.draw_detections(frame, detections)
        
        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 显示FPS
        cv2.putText(result_frame, f"FPS: {int(fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Detection", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()