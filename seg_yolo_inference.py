import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

THRESHOLD = 0.5

segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/SegFormer/checkpoints/segformer_epoch5.pth", map_location="cpu")
segformer_model.load_state_dict(checkpoint)
segformer_model.eval().cuda()

segformer_processor = SegformerImageProcessor(do_resize=True, size=512, do_normalize=True)
yolo_model = YOLO("runs/detect/train/weights/best.pt")

img_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/WhatsApp Image 2025-05-01 at 00.58.52_5a8fd4ba.jpg"
orig_img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

seg_input = segformer_processor(images=img_rgb, return_tensors="pt").to("cuda")
with torch.no_grad():
    seg_out = segformer_model(**seg_input)

probs = torch.softmax(seg_out.logits, dim=1)
road_prob = probs[:, 1, :, :]
seg_mask = (road_prob > THRESHOLD).squeeze().cpu().numpy().astype(np.uint8)
seg_mask_resized = cv2.resize(seg_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

results = yolo_model.predict(img_path, conf=0.3, save=False)[0]
boxes = results.boxes.xyxy.cpu().numpy()
scores = results.boxes.conf.cpu().numpy()
classes = results.boxes.cls.cpu().numpy()

mask_colored = np.zeros_like(orig_img)
mask_colored[seg_mask_resized == 1] = (0, 255, 0)
overlay = cv2.addWeighted(orig_img, 0.7, mask_colored, 0.3, 0)

for box, score, cls in zip(boxes, scores, classes):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(overlay, f"{yolo_model.names[int(cls)]} {score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("combined_output.jpg", overlay)
print("Saved to combined_output.jpg")
