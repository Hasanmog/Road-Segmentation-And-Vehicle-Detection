import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from full_encoder import MultiScaleFusion  


img_path = "12.png"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]

ckpt_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/Encoder/img_encoder/weights/sam_vit_b_01ec64.pth"

model = MultiScaleFusion(
    img_size=512,
    small_patch_size=8,
    large_patch_size=16,
    enc_ckpt_path=ckpt_path,
    backbone_freeze=True,
    dim=(256, 256),
    depth=(2, 2),
    num_heads=(4, 4),
    mlp_ratio=(4, 4)
)
model.eval()

# === Forward pass ===
with torch.no_grad():
    fused_features = model(image_tensor)  # List of [B, N, C]

def visualize_tokens_as_feature_maps(tokens, title):
    B, N, C = tokens.shape
    sqrt = int(N ** 0.5)
    if sqrt * sqrt != N:
        print(f"Cannot reshape {N} tokens into square (H, W) — skipping.")
        return

    feat = tokens[0].transpose(0, 1).reshape(C, sqrt, sqrt)  # [C, H, W]

    channels_to_plot = min(10, feat.shape[0])
    fig, axes = plt.subplots(1, channels_to_plot, figsize=(15, 5))
    fig.suptitle(title)

    for i in range(channels_to_plot):
        ax = axes[i]
        ax.imshow(feat[i].cpu(), cmap='plasma')  # Try 'inferno', 'magma', etc.
        ax.axis('off')
        ax.set_title(f'Channel {i}')

    plt.tight_layout()
    plt.show()


visualize_tokens_as_feature_maps(fused_features[0], "Fused Local Tokens")
visualize_tokens_as_feature_maps(fused_features[1], "Fused Global Tokens")


def overlay_on_image(feature_map, image_tensor, alpha=0.5, title="Overlayed Feature Map"):
    feature_map_resized = F.interpolate(
        feature_map.unsqueeze(0).unsqueeze(0),
        size=(512, 512),
        mode='bilinear'
    )[0, 0]
    feature_map_resized = (feature_map_resized - feature_map_resized.min()) / (feature_map_resized.max() - feature_map_resized.min())

    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.imshow(feature_map_resized.cpu(), cmap='inferno', alpha=alpha)
    plt.axis('off')
    plt.title(title)
    plt.show()


with torch.no_grad():
    B, N, C = fused_features[0].shape
    sqrt = int(N ** 0.5)
    if sqrt * sqrt == N:
        feat_map = fused_features[0][0, :, 0].reshape(sqrt, sqrt)  
        overlay_on_image(feat_map, image_tensor, title="Overlay: Local Feature Channel 0")


