# script for evaluating the whole pipeline
import torch
import numpy as np
import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import yolo_predict, sam_predict, load_sam, load_yolo
from utils.visualize import display_results , visualize_boxes , visualize_masks 

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "/home/hasanmog/AUB_Masters/projects/Joint-Detection-and-Segmentation-Framework-for-Landmarks/weights/sam/sam_vit_h_4b8939.pth"
yolo_checkpoint = "/home/hasanmog/AUB_Masters/projects/Joint-Detection-and-Segmentation-Framework-for-Landmarks/weights/yolo/yolo11n.pt"
image_path = "/home/hasanmog/AUB_Masters/projects/Joint-Detection-and-Segmentation-Framework-for-Landmarks/imgs/01-arches-of-time-the-modern-landmark-london-rezvan-yarhaghi.jpg"

image_bgr = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                         
sam = load_sam(sam_checkpoint)
yolo = load_yolo(yolo_checkpoint)


boxes_raw, confidences, classes = yolo_predict(yolo, image=image_rgb)

masks, boxes = sam_predict(predictor=sam, image=image_rgb, boxes=boxes_raw)

display_results(image_bgr, boxes, masks, confidences, classes)

img_with_boxes = visualize_boxes(image_rgb, boxes, confidences, classes)
img_with_masks = visualize_masks(image_rgb, masks)

cv2.imwrite('output_boxes.jpg', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
cv2.imwrite('output_masks.jpg', cv2.cvtColor(img_with_masks, cv2.COLOR_RGB2BGR))


