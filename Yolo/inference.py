import cv2
from ultralytics import YOLO
import os

model = YOLO("runs/detect/vehicle_yolo/weights/best.pt") 

image_path = "/home/hasanmog/datasets/vedai/vedai_yolo/images/test/some_image.jpg"  
results = model.predict(source=image_path, save=True, save_txt=True)


output_path = os.path.join("runs/detect/predict", os.path.basename(image_path))
img = cv2.imread(output_path)
cv2.imshow("YOLOv8 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
