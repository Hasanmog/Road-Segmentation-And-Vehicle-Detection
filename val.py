import torch
import torch.nn as nn
import torchmetrics
import json
import os
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from criterion import det_criterion, seg_criterion

# --- Helper function ---
def box_iou_single(boxes1, boxes2):
    """
    boxes1: [1, 4]
    boxes2: [N, 4]
    Returns IoU [1, N]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1 + area2 - inter

    return inter / union  # shape [1, N]


@torch.no_grad()
def validate_one_epoch(model, seg_loader, det_loader, device, run=None, args=None):
    model.eval()

    seg_loss_total = 0
    det_loss_total = 0

    iou_metric = torchmetrics.JaccardIndex(task="binary", num_classes=1).to(device)
    detection_metric = MeanAveragePrecision().to(device)
    accuracy_metric_seg = torchmetrics.classification.BinaryAccuracy().to(device)
    accuracy_metric_det = torchmetrics.classification.MulticlassAccuracy(num_classes=2).to(device)


    bbox_iou_metric = torchmetrics.IoU(task="binary", num_classes=1).to(device)
    classification_correct = 0
    classification_total = 0

    num_batches = min(len(seg_loader), len(det_loader))
    seg_iter = iter(seg_loader)
    det_iter = iter(det_loader)

    for _ in tqdm(range(num_batches)):
        seg_batch = next(seg_iter)
        det_batch = next(det_iter)

        seg_img, gt_mask = seg_batch
        det_img, target = det_batch

        seg_img, gt_mask = seg_img.to(device), gt_mask.to(device)
        det_img = det_img.to(device)
        gt_cls = target["cls"].to(device)
        gt_box = target["bbox"].to(device)
        gt_center = target["centerness"].to(device)

        with torch.autocast(device_type=device):
            # ---- Segmentation Part ----
            output = model(seg_img)
            class_logits = output["mask_logits"]
            masks = output["masks"]

            best_queries = class_logits.squeeze(-1).sigmoid().max(dim=1)[1]
            batch_size = masks.size(0)
            selected_masks = torch.stack([masks[b, best_queries[b]] for b in range(batch_size)], dim=0)
            selected_masks = selected_masks.unsqueeze(1)

            seg_loss = seg_criterion(selected_masks, gt_mask)
            seg_loss_total += seg_loss.item()

            preds = torch.sigmoid(selected_masks) > 0.5
            iou_metric.update(preds, gt_mask.int())
            accuracy_metric_seg.update(preds.int(), gt_mask.int())

            # ---- Detection Part ----
            output = model(det_img)
            pred_box = output["bbox"]
            pred_label = output["cls_logits"]
            pred_center = output["centerness"]

            preds_det = []
            targets_det = []

            for b in range(pred_box.size(0)):
                stride = (args.img_size // 8) if args else 64  # or match model stride

                boxes = pred_box[b].permute(1, 2, 0).reshape(-1, 4) * stride
                scores = pred_center[b].permute(1, 2, 0).reshape(-1)

                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                labels = pred_label[b].softmax(dim=0)
                labels = labels.permute(1, 2, 0).reshape(-1, 2)
                labels = labels.argmax(dim=-1)

                mask = scores > 0.5  # threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                preds_det.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })

                # Ground truth
                boxes_gt = gt_box[b].permute(1, 2, 0).reshape(-1, 4) * stride
                cx_gt, cy_gt, w_gt, h_gt = boxes_gt[:, 0], boxes_gt[:, 1], boxes_gt[:, 2], boxes_gt[:, 3]
                x1_gt = cx_gt - w_gt / 2
                y1_gt = cy_gt - w_gt / 2
                x2_gt = cx_gt + w_gt / 2
                y2_gt = cy_gt + w_gt / 2
                boxes_gt = torch.stack([x1_gt, y1_gt, x2_gt, y2_gt], dim=-1)

                labels_gt = gt_cls[b].reshape(-1)

                targets_det.append({
                    "boxes": boxes_gt,
                    "labels": labels_gt,
                })

                # Update box IoU and classification accuracy
                if len(boxes) > 0 and len(boxes_gt) > 0:
                    for pred_idx in range(boxes.size(0)):
                        pred_box_single = boxes[pred_idx].unsqueeze(0)
                        pred_label_single = labels[pred_idx]

                        ious = box_iou_single(pred_box_single, boxes_gt)
                        max_iou, max_idx = ious.max(dim=1)

                        bbox_iou_metric.update((pred_box_single, boxes_gt[max_idx]))

                        if max_iou.item() > 0.5:
                            gt_label = labels_gt[max_idx]
                            if pred_label_single == gt_label:
                                classification_correct += 1
                            classification_total += 1

            detection_metric.update(preds_det, targets_det)

            # Detection classification loss (raw prediction grid)
            pred_logits = pred_label.permute(0, 2, 3, 1).reshape(-1, 2)
            gt_labels_flat = gt_cls.view(-1)
            valid = (gt_labels_flat >= 0)
            if valid.sum() > 0:
                accuracy_metric_det.update(pred_logits.softmax(dim=-1)[valid], gt_labels_flat[valid])

            # Detection loss
            det_loss, _ = det_criterion([pred_label, pred_box, pred_center], [gt_cls, gt_box, gt_center])
            det_loss_total += det_loss.item()

    # ---- After Validation ----
    mean_iou = iou_metric.compute()
    mean_ap = detection_metric.compute()["map"].item()
    segmentation_acc = accuracy_metric_seg.compute()
    detection_acc = accuracy_metric_det.compute()
    mean_bbox_iou = bbox_iou_metric.compute()

    if classification_total > 0:
        box_classification_accuracy = classification_correct / classification_total
    else:
        box_classification_accuracy = 0.0

    avg_seg_loss = seg_loss_total / num_batches
    avg_det_loss = det_loss_total / num_batches

    # Logging to Neptune
    if run:
        run["validation/seg_loss"].append(avg_seg_loss)
        run["validation/det_loss"].append(avg_det_loss)
        run["validation/miou"].append(mean_iou)
        run["validation/segmentation_accuracy"].append(segmentation_acc)
        run["validation/detection_accuracy"].append(detection_acc)
        run["validation/bounding_box_iou"].append(mean_bbox_iou)
        run["validation/box_classification_accuracy"].append(box_classification_accuracy)
        run["validation/map"].append(mean_ap)

    # Save + Print
    results = {
        "segmentation_loss": avg_seg_loss,
        "detection_loss": avg_det_loss,
        "mean_iou": mean_iou.item(),
        "segmentation_accuracy": segmentation_acc.item(),
        "detection_accuracy": detection_acc.item(),
        "bounding_box_iou": mean_bbox_iou.item(),
        "box_classification_accuracy": box_classification_accuracy,
        "mean_average_precision": mean_ap
    }
    print("Validation Results:", results)

    if args:
        os.makedirs(args.out_dir, exist_ok=True)
        json_path = os.path.join(args.out_dir, "validation_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

    model.train()

    return avg_seg_loss, avg_det_loss, mean_iou, mean_ap
