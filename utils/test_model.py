import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from model import SegDet


# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = '/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/weights/sam_vit_b_01ec64.pth'  
model = SegDet(ckpt_path=ckpt_path).to(device)
model.eval()

# --- Load & preprocess image ---
img_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/12.png"  # replace with your image path
image = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 512, 512]

# --- Forward pass ---
with torch.no_grad():
    output = model(input_tensor)

# --- Post-process segmentation mask ---
mask = output['masks'].sigmoid().squeeze().cpu().numpy()  # [512, 512]
mask_overlay = (mask > 0.5).astype(np.uint8)  # binary mask

# --- Post-process detections ---
bbox = output['bbox'].squeeze().cpu()         # [4, H, W]
obj_score = output['obj_score'].squeeze().cpu()  # [H, W]
class_score = output['class_score'].squeeze().cpu()  # [H, W]
print("class_score" , class_score)
print("obj_score" , obj_score)
# --- Get detections from grid ---
threshold = 0.5
H, W = obj_score.shape
stride_y = 512 / H
stride_x = 512 / W

detections = []
for i in range(H):
    for j in range(W):
        if obj_score[i, j] > threshold:
            x_center = (j + bbox[0, i, j]) * stride_x
            y_center = (i + bbox[1, i, j]) * stride_y
            width    = bbox[2, i, j] * 512
            height   = bbox[3, i, j] * 512

            x = x_center - width / 2
            y = y_center - height / 2
            conf = obj_score[i, j].item()
            detections.append([x, y, width, height, conf])

# --- Plot ---
fig, ax = plt.subplots(1, figsize=(8, 8))
image_vis = np.array(image.resize((512, 512)))

# Overlay mask
ax.imshow(image_vis)
ax.imshow(mask_overlay, cmap='viridis', alpha=0.4)

# Plot bounding boxes
for det in detections:
    x, y, w, h, score = det
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, f'vehicle: {score:.2f}', color='white', fontsize=8, bbox=dict(facecolor='blue', alpha=0.5))

ax.axis('off')
plt.title("Segmentation + Detection Overlay")
plt.tight_layout()
plt.show()


