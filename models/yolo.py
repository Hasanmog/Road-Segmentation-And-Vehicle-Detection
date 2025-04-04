from ultralytics import YOLO
import numpy as np
import torch


def load_yolo(weights_path: str):
    
    model = YOLO(weights_path)
    return model


def yolo_predict(model, image: np.ndarray, conf_threshold: float = 0.5):

    results = model(image, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    filtered_boxes = []
    filtered_confs = []
    filtered_classes = []

    for box, conf, cls in zip(boxes, confidences, class_ids):
        if conf >= conf_threshold:
            filtered_boxes.append(box.tolist())
            filtered_confs.append(float(conf))
            filtered_classes.append(int(cls))

    return filtered_boxes, filtered_confs, filtered_classes
