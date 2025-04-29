# inference_on_image.py

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse
from model.model import SegDet

# Helper Functions
def plot_segmentation(img, pred_mask):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.imshow(pred_mask.squeeze(0).cpu(), alpha=0.5, cmap='jet')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()

def plot_detection(img, boxes, scores, labels, score_thresh=0.5):
    plt.figure(figsize=(8, 8))
    plt.imshow(img.permute(1, 2, 0).cpu())
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                          edgecolor='lime', facecolor='none', linewidth=2))
        plt.text(x1, y1, f"{label.item()}:{score:.2f}", color='yellow', fontsize=8)
    plt.title('Detection Results')
    plt.axis('off')
    plt.show()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Image
    image = Image.open(args.image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0).to(device)

    # Load Model
    model = SegDet(
        img_size=args.img_size,
        small_patch_size=args.small_patch,
        large_patch_size=args.large_patch,
        backbone=args.backbone,
        sam_ckpt_path=args.sam_ckpt,
        swin_det_path=args.swin_det_ckpt,
        swin_seg_path=args.swin_seg_ckpt,
        backbone_freeze=True
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(img)

    # ---- Segmentation Visualization ----
    class_logits = output["mask_logits"]
    masks = output["masks"]

    best_queries = class_logits.squeeze(-1).sigmoid().max(dim=1)[1]
    selected_masks = torch.stack([masks[0, best_queries[0]]], dim=0)
    selected_masks = selected_masks.unsqueeze(1)

    plot_segmentation(img.squeeze(0), selected_masks.squeeze(0) > 0.5)

    # ---- Detection Visualization ----
    pred_box = output["bbox"][0]
    pred_label = output["cls_logits"][0]
    pred_center = output["centerness"][0]

    stride = (args.img_size // 8)  # 512/8 = 64 if you use that setting

    boxes = pred_box.permute(1, 2, 0).reshape(-1, 4) * stride
    scores = pred_center.permute(1, 2, 0).reshape(-1)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    labels = pred_label.softmax(dim=0)
    labels = labels.permute(1, 2, 0).reshape(-1, 2)
    labels = labels.argmax(dim=-1)

    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    plot_detection(img.squeeze(0), boxes, scores, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to your input image')

    # Model configs
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--small_patch', type=int, default=8)
    parser.add_argument('--large_patch', type=int, default=16)
    parser.add_argument('--backbone', type=str, default="SWIN")
    parser.add_argument('--sam_ckpt', type=str, default=None)
    parser.add_argument('--swin_det_ckpt', type=str, default=None)
    parser.add_argument('--swin_seg_ckpt', type=str, default=None)

    args = parser.parse_args()
    main(args)
