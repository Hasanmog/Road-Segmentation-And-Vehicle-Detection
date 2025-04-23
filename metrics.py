import torch

def mask_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()
    
    intersection = (pred_bin * gt_bin).flatten(1).sum(dim=1)
    union = pred_bin.flatten(1).sum(dim=1) + gt_bin.flatten(1).sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes in (cx, cy, w, h) format.
    box1: Tensor of shape [N, 4] - predicted boxes
    box2: Tensor of shape [N, 4] - ground truth boxes
    Returns: Tensor of shape [N] with IoU values
    """

    box1_x1 = box1[:, 0] - box1[:, 2] / 2
    box1_y1 = box1[:, 1] - box1[:, 3] / 2
    box1_x2 = box1[:, 0] + box1[:, 2] / 2
    box1_y2 = box1[:, 1] + box1[:, 3] / 2

    box2_x1 = box2[:, 0] - box2[:, 2] / 2
    box2_y1 = box2[:, 1] - box2[:, 3] / 2
    box2_x2 = box2[:, 0] + box2[:, 2] / 2
    box2_y2 = box2[:, 1] + box2[:, 3] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = area1 + area2 - inter_area + 1e-6

    iou = inter_area / union_area
    return iou




def class_acc(pred_logits, gt_labels, obj_mask, threshold=0.5):
    """
    pred_logits: raw class logits (before sigmoid) – shape [B, H, W]
    gt_labels: ground truth class labels – shape [B, H, W]
    obj_mask: binary mask indicating object presence – shape [B, H, W]
    """
    with torch.no_grad():

        pred_labels = (pred_logits > threshold).long()

        
        mask = obj_mask.bool()
        if mask.sum() == 0:
            return 0.0 

        correct = (pred_labels[mask] == gt_labels[mask]).sum()
        total = mask.sum()

        accuracy = correct.float() / total.float()
        return accuracy.item()


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    prob = torch.sigmoid(pred)
    pt = prob * target + (1 - prob) * (1 - target)  # p_t
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()
