import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from full_encoder import MultiScaleFusion

# === Load & Preprocess Image ===
img_path = "12.png"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Ensure input matches model expectations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]

# === Initialize Full MultiScaleFusion Model ===
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

# === Check Trainable Params ===
trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
if trainable_params:
    print("Trainable parameters:")
    for name in trainable_params:
        print(f"  - {name}")
else:
    print("✅ All model parameters are frozen.")

# === Forward Pass ===
with torch.no_grad():
    fused_features = model(image_tensor)  # List of two tensors, one per branch

# === Output Shapes ===
for i, feat in enumerate(fused_features):
    print(f"Fused Output {i} Shape: {feat.shape}")  # Should be [B, N_patches, C]

# === Visualize Some Tokens (reshaped to feature maps) ===
def visualize_tokens_as_feature_maps(tokens, title):
    """
    tokens: [B, N, C] (e.g. [1, 4095, 256])
    Will auto-infer H and W from N (should be square)
    """
    B, N, C = tokens.shape
    sqrt = int(N ** 0.5)
    if sqrt * sqrt != N:
        print(f"⚠️ Cannot reshape {N} tokens into square (H, W) — skipping.")
        return

    feat = tokens[0].transpose(0, 1).reshape(C, sqrt, sqrt)  # [C, H, W]

    channels_to_plot = min(8, feat.shape[0])
    fig, axes = plt.subplots(1, channels_to_plot, figsize=(15, 5))
    fig.suptitle(title)

    for i in range(channels_to_plot):
        ax = axes[i]
        ax.imshow(feat[i].cpu(), cmap='plasma')
        ax.axis('off')
        ax.set_title(f'Channel {i}')

    plt.tight_layout()
    plt.show()
    
# Assume image size 512×512, patches 8×8 → 64×64, patches 16×16 → 32×32
visualize_tokens_as_feature_maps(fused_features[0], "Fused Local Tokens")
visualize_tokens_as_feature_maps(fused_features[1], "Fused Global Tokens")




