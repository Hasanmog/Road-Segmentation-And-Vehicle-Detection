# infer_segformer.py
import os
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load("checkpoints/segformer_epoch5.pth"))
model.to(DEVICE)
model.eval()

def infer_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    tensor_image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=tensor_image)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        preds = torch.softmax(logits, dim=1)[:, 1, :, :]
        pred_mask = (preds > 0.3).squeeze().cpu().numpy() * 255
        pred_mask = pred_mask.astype("uint8")

    return image, pred_mask

def visualize(image, mask):
    mask_image = Image.fromarray(mask)
    mask_overlay = np.array(image).copy()
    mask_colored = np.stack([mask // 2, mask, mask // 2], axis=2)  # green mask
    mask_overlay[mask > 0] = mask_colored[mask > 0]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    axs[2].imshow(mask_overlay)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_image = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/00000807_jpg.rf.d02f795b274c9001b5cf07d32945a5a7.jpg"
    image, mask = infer_image(input_image)
    visualize(image, mask)
