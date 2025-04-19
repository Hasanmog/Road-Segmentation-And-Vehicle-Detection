import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from model.model import SegDet

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegDet(ckpt_path=None).to(device)
checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_1.pt")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Load & preprocess image ---
img_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/utils/00001231_jpg.rf.0d98380a69d5966446d60b21f6284f03.jpg"
image = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)

# --- Forward pass ---
with torch.no_grad():
    output = model(input_tensor)
# --- Post-process segmentation mask ---
mask = output['masks'].sigmoid().squeeze().cpu().numpy()
print(mask)
mask_overlay = (mask > 0.3).astype(np.uint8)

# --- Post-process detections ---
bbox = output['bbox'].squeeze().cpu()           # [4, H, W]
obj_score = torch.sigmoid(output['obj_score'].squeeze().cpu())  # [H, W]
class_score = torch.sigmoid(output['class_score'].squeeze().cpu())  # [H, W]

H, W = obj_score.shape
stride_y = 512 / H
stride_x = 512 / W

threshold = 0.02
detections = []
for i in range(H):
    for j in range(W):
        if obj_score[i, j] > threshold:
            x1 = bbox[0, i, j].item() * 512
            y1 = bbox[1, i, j].item() * 512
            x2 = bbox[2, i, j].item() * 512
            y2 = bbox[3, i, j].item() * 512

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            conf = obj_score[i, j].item()
            label_score = class_score[i, j].item()
            detections.append([x1, y1, x2, y2, conf, label_score])

# --- Plotting ---
image_vis = np.array(image.resize((512, 512)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Segmentation
ax1.imshow(image_vis)
ax1.imshow(mask, cmap='magma', alpha=0.6)
ax1.set_title("Segmentation Mask Overlay")
ax1.axis('off')

# Detections
ax2.imshow(image_vis)
for det in detections:
    x1, y1, x2, y2, conf, label_score = det
    w = x2 - x1
    h = y2 - y1
    label = "Truck" if label_score > 0.5 else "Car"
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='blue', facecolor='none')
    ax2.add_patch(rect)
    ax2.text(x1, y1, f'{label} ({conf:.2f})', color='white', fontsize=8,
             bbox=dict(facecolor='blue', alpha=0.5))

ax2.set_title("Vehicle Detections")
ax2.axis('off')

plt.tight_layout()
plt.show()
