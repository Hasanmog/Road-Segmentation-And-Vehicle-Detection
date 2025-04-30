import torch

def postprocess_bbox(bbox, img_h, img_w):
    B, _, H, W = bbox.shape
    boxes = torch.zeros_like(bbox)

    for b in range(B):
        for i in range(H):
            for j in range(W):
                cx = bbox[b, 0, i, j] * img_w
                cy = bbox[b, 1, i, j] * img_h
                w = bbox[b, 2, i, j] * img_w
                h = bbox[b, 3, i, j] * img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes[b, :, i, j] = torch.tensor([x1, y1, x2, y2])
    return boxes


