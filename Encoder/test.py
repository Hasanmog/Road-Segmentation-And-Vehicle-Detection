import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from full_encoder import MultiScaleFusion  

# === Load & Preprocess Image ===
img_path = "12.png"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]

# === Load Encoder ===
ckpt_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/Encoder/img_encoder/weights/sam_vit_b_01ec64.pth"

model = MultiScaleFusion(
    img_size=512,
    small_patch_size=8,
    large_patch_size=16,
    enc_ckpt_path=ckpt_path,
    backbone_freeze=True,
    dim=(256,256),
    depth=(2, 2),
    num_heads=(4, 4),
    mlp_ratio=(4, 4)
)
model.eval()

# === Run Model ===
with torch.no_grad():
    fused_features = model(image_tensor)  # List of [B, N, C]

# === Extract First Feature Map ===
tokens = fused_features[0]  # Local tokens (for example)
B, N, C = tokens.shape
sqrt = int(N ** 0.5)

if sqrt * sqrt == N:
    # Reshape one channel into [H, W]
    feat_map = tokens[0, :, 0].reshape(sqrt, sqrt)

    # === Overlay Feature Map ===
    def overlay_on_image(feature_map, image_tensor, alpha=0.5, title="Overlayed Feature Map"):
        feature_map_resized = F.interpolate(
            feature_map.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=True
        )[0, 0]
        feature_map_resized = (feature_map_resized - feature_map_resized.min()) / (feature_map_resized.max() - feature_map_resized.min())

        img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.imshow(feature_map_resized.cpu(), cmap='inferno', alpha=alpha)
        plt.axis('off')
        plt.title(title)
        plt.show()

    overlay_on_image(feat_map, image_tensor, title="Overlay: Feature Channel 0")

    # === Print Feature Map Info ===
    print(f"Feature Map Shape: {tokens.shape}")  # [B, N, C]
    print(f"Feature Map Resolution: {sqrt}x{sqrt}")
    print(f"Total Channels: {C}")
else:
    print("Cannot reshape feature map into a square for visualization.")
