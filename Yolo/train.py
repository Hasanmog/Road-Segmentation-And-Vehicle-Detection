from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/home/hasanmog/datasets/vedai/vedai_yolo/vedai.yaml", epochs=50, imgsz=512, batch=16)

