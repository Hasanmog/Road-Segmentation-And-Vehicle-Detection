import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from model.model import SegDet
# --- Model Loading ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegDet(ckpt_path=None).to(device)
checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_1.pt")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Image Loading ---
img_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/utils/00001231_jpg.rf.0d98380a69d5966446d60b21f6284f03.jpg"
image = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)

# --- Inference ---
with torch.no_grad():
    output = model(input_tensor)

mask = output["masks"]
box = output["bbox"]
obj_score = output["obj_score"]
class_score = output["class_score"]


print(f"mask shape: {mask.shape}")              # Expected: [1, 1, 512, 512]
print(f"bbox shape: {box.shape}")               # Expected: [1, 4, 64, 64]
print(f"obj_score shape: {obj_score.shape}")    # Expected: [1, 64, 64]
print(f"class_score shape: {class_score.shape}")# Expected: [1, 64, 64]

# Optional: squeeze batch dimension
mask = mask.squeeze(0)
box = box.squeeze(0)
obj_score = obj_score.squeeze(0)
class_score = class_score.squeeze(0)

# Get confident predictions
conf_thresh = 0.5
obj_cells = (obj_score > conf_thresh).nonzero(as_tuple=False)

print(f"\nTotal confident detections (obj > {conf_thresh}): {len(obj_cells)}")
for idx, (i, j) in enumerate(obj_cells[:3]):  # just show top 3
    print(f"\n--- Detection {idx+1} at cell ({i.item()}, {j.item()}) ---")
    print("Objectness score:", obj_score[i, j].item())
    print("Predicted class (argmax):", class_score[i, j].item())
    print("Predicted box (cx, cy, w, h):", box[:, i, j])






