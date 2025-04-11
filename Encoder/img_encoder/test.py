import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from encoder import Image_Encoders



img_path = "12.png"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)


ckpt_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/Encoder/img_encoder/weights/sam_vit_b_01ec64.pth"
model = Image_Encoders(ckpt_path=ckpt_path, freeze=True)
model.eval()


trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
if trainable_params:
    print("Trainable parameters:")
    for name in trainable_params:
        print(f"  - {name}")
else:
    print("✅ All model parameters are frozen.")


with torch.no_grad():
    local, global_feat = model(image_tensor)

print(f"Local shape: {local.shape}, Global shape: {global_feat.shape}")


def visualize_feature_map(feat, title):
    feat = feat.squeeze(0)
    channels_to_plot = min(4, feat.shape[0])
    fig, axes = plt.subplots(1, channels_to_plot, figsize=(15, 5))
    fig.suptitle(title)

    for i in range(channels_to_plot):
        ax = axes[i]
        ax.imshow(feat[i].cpu(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i}')

    plt.tight_layout()
    plt.show()

visualize_feature_map(local, "Local Features (Small Patch Encoder)")
visualize_feature_map(global_feat, "Global Features (Large Patch Encoder)")


