# -*- coding: utf-8 -*-
import os

# Verify the path
print(os.environ['LD_LIBRARY_PATH'])
import cv2
import numpy as np
from IPython.display import display, clear_output,Image

from time import time
from ais_bench.infer.interface import InferSession
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('fire.yaml'))['names']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = InferSession(device_id=0, model_path="best_1_inoutfp32.om")

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(original_image):
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    cv2.imshow("image",image)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    
    begin_time = time()
    outputs = model.infer(feeds=blob, mode="static")
    print(outputs)
    end_time = time()
    print("om infer time:", end_time - begin_time)

    outputs = np.array([cv2.transpose(outputs[0][0])])
    print(outputs)
    rows = outputs.shape[1]
    print(rows)
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        print(classes_scores)
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.05:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    print(boxes,scores)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    print(result_boxes)
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
    cv2.imshow("detect",original_image)
    cv2.waitKey(0)
                          
image = cv2.imread("./test_images/1.jpg")
image = cv2.resize(image,(640,640))
main(image)
# # Initialize the camera
# camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# # Set the codec to MJPG if it is supported
# if camera.isOpened():
#     # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
#     # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
#     camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# else:
#     raise IOError("Cannot open the webcam")

#     # Define the codec and create VideoWriter object
# # Get the width and height of the frames
# frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"Frame width: {frame_width}, Frame height: {frame_height}")

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame_width, frame_height))  # 20.0 is the frame rate

# try:
#     _start_time = time()
#     while time() - _start_time < 5:
#         # Capture frame-by-frame
#         ret, frame = camera.read()
#         if not ret:
#             raise IOError("Cannot capture frame")
#         main(frame)
#         out.write(frame)

#         # Display the image
#         # clear_output(wait=True)
        
#         # # Afficher l'image capturée
#         # display(Image(data=cv2.imencode('.jpg', frame)[1]))

# finally:
#     # When everything done, release the capture
#     camera.release()
#     out.release()
