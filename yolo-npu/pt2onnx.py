from ultralytics import YOLO


model = YOLO("yolov8n.pt")  

# Export the model
model.export(format="onnx",opset=12)