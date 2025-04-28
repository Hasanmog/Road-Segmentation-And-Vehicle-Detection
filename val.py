import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchmetrics
import json
import os
from criterion import det_criterion, seg_criterion
from data.dataloader import RoadSegDataset , VehicleDetDataset
from torch.utils.data import DataLoader
from model.model import SegDet

@torch.no_grad()
def validate_one_epoch(model, seg_loader, det_loader, device, run=None, args=None):
    model.eval()

    seg_loss_total = 0
    det_loss_total = 0

    iou_metric = torchmetrics.JaccardIndex(task="binary", num_classes=1).to(device)
    detection_metric = MeanAveragePrecision().to(device)

    num_batches = min(len(seg_loader), len(det_loader))

    seg_iter = iter(seg_loader)
    det_iter = iter(det_loader)

    for _ in range(num_batches):
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
            output = model(seg_img)
            class_logits = output["mask_logits"]
            masks = output["masks"]

            best_queries = class_logits.squeeze(-1).sigmoid().max(dim=1)[1]
            batch_size = masks.size(0)
            selected_masks = torch.stack([masks[b, best_queries[b]] for b in range(batch_size)], dim=0)

            selected_masks = selected_masks.unsqueeze(1)  # [B, 1, 512, 512]

            seg_loss = seg_criterion(selected_masks, gt_mask)
            seg_loss_total += seg_loss.item()

            preds = torch.sigmoid(selected_masks) > 0.5
            iou_metric.update(preds, gt_mask.int())

            output = model(det_img)
            pred_box = output["bbox"]
            pred_label = output["cls_logits"]
            pred_center = output["centerness"]

            preds_det = []
            targets_det = []

            for b in range(pred_box.size(0)):
                boxes = pred_box[b].permute(1, 2, 0).reshape(-1, 4)
                scores = pred_center[b].permute(1, 2, 0).reshape(-1)

                labels = pred_label[b].softmax(dim=0)
                labels = labels.permute(1, 2, 0).reshape(-1, 2)
                labels = labels.argmax(dim=-1)  # [4096] with 0 or 1
                
                mask = scores > 0.5
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                preds_det.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })


                boxes_gt = gt_box[b].permute(1, 2, 0).reshape(-1, 4)
                labels_gt = gt_cls[b].reshape(-1)

                mask_gt = labels_gt >= 0

                targets_det.append({
                    "boxes": boxes_gt[mask_gt],
                    "labels": labels_gt[mask_gt],
                })


            detection_metric.update(preds_det, targets_det)

            det_loss, _ = det_criterion([pred_label, pred_box, pred_center], [gt_cls, gt_box, gt_center])
            det_loss_total += det_loss.item()

    mean_iou = iou_metric.compute()
    map_result = detection_metric.compute()
    mean_ap = map_result["map"].item()

    avg_seg_loss = seg_loss_total / num_batches
    avg_det_loss = det_loss_total / num_batches

    if run:
        run["validation/seg_loss"].append(avg_seg_loss)
        run["validation/det_loss"].append(avg_det_loss)
        run["validation/miou"].append(mean_iou)
        run["validation/map"].append(mean_ap)
        
    results = {
            "segmentation_loss": avg_seg_loss,
            "detection_loss": avg_det_loss,
            "mean_iou": mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou,
            "mean_average_precision": mean_ap
        }
    print("results" , results)

    if args:
        os.makedirs(args.out_dir, exist_ok=True)
        json_path = os.path.join(args.out_dir, "validation_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

    model.train()

    return avg_seg_loss, avg_det_loss, mean_iou, mean_ap


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from model.model import SegDet
    from data.dataloader import RoadSegDataset, VehicleDetDataset
    from val import validate_one_epoch

    if __name__ == "__main__":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        seg_dataset = RoadSegDataset(dataset_dir="/home/hasanmog/datasets/dataset_reduced", mode='val', img_size=512)
        det_dataset = VehicleDetDataset(dataset_dir="/home/hasanmog/datasets/vedai", mode='val', img_size=512, grid_size=64)

        seg_loader = DataLoader(seg_dataset, batch_size=2, shuffle=False, num_workers=0)
        det_loader = DataLoader(det_dataset, batch_size=2, shuffle=False, num_workers=0)

        model = SegDet(
            img_size=512,
            small_patch_size=8,
            large_patch_size=16,
            backbone="SAM",
            sam_ckpt_path=None,
            swin_det_path=None,
            swin_seg_path=None,
            backbone_freeze=True
        )
        model.to(device)

        # Load checkpoint
        checkpoint_path = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_1.pt"  # <<< PUT your path here
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        print("Checkpoint loaded. Running validation test...")
        validate_one_epoch(model, seg_loader, det_loader, device, run=None, args=None)
