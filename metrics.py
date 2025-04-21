import torch

def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0.5).float()
    gt_bin = (gt_mask > 0.5).float()
    
    intersection = (pred_bin * gt_bin).flatten(1).sum(dim=1)
    union = pred_bin.flatten(1).sum(dim=1) + gt_bin.flatten(1).sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def iou(box1, box2):
    """
    Compute IoU between two sets of boxes in (x1, y1, x2, y2) format.
    box1: Tensor of shape [N, 4]
    box2: Tensor of shape [N, 4]
    Returns: Tensor of shape [N] with IoU values
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1 + area2 - inter_area + 1e-6
    iou = inter_area / union
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
