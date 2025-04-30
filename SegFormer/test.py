import numpy as np
import torch
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from data.dataloader import RoadSegDataset


def iou(preds, labels, pos_label=1):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    thresholds = [0.3, 0.5, 0.75]
    iou_scores = []
    for threshold in thresholds:
        preds_binary = np.array([(pred > threshold).astype(np.float32) for pred in preds])
        labels_binary = np.array([(label == pos_label).astype(np.float32) for label in labels])
        intersection = np.sum(preds_binary * labels_binary)
        union = np.sum(preds_binary) + np.sum(labels_binary) - intersection
        if np.count_nonzero(union) == 0:
            iou_scores.append(1.0 if np.count_nonzero(intersection) == 0 else 0.0)
        else:
            iou_scores.append(intersection / union)
    return tuple(iou_scores)


def calc_f1_score(predicted_masks, gt_masks, threshold=0.5):
    predicted_masks = predicted_masks.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()
    predicted_masks = (predicted_masks > threshold).astype(np.uint8).flatten()
    gt_masks = (gt_masks > 0.5).astype(np.uint8).flatten()
    precision = precision_score(gt_masks, predicted_masks, average='binary', zero_division=1)
    recall = recall_score(gt_masks, predicted_masks, average='binary', zero_division=1)
    f1 = f1_score(gt_masks, predicted_masks, average='binary', zero_division=1)
    return precision, recall, f1


def test_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    IMG_SIZE = 512

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/checkpoints/segformer_epoch5.pth"))
    model.to(DEVICE)
    model.eval()

    test_dataset = RoadSegDataset("/home/hasanmog/datasets/dataset_zoom_original/dataset_zoom_original/", IMG_SIZE, "test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_iou_30, total_iou_50, total_iou_75 = 0, 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.softmax(logits, dim=1)[:, 1, :, :]

            iou30, iou50, iou75 = iou(preds, labels)
            precision, recall, f1 = calc_f1_score(preds, labels)

            total_iou_30 += iou30
            total_iou_50 += iou50
            total_iou_75 += iou75
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

    print("\nTest Results:")
    print(f"IoU@0.3: {total_iou_30 / count:.4f}, IoU@0.5: {total_iou_50 / count:.4f}, IoU@0.75: {total_iou_75 / count:.4f}")
    print(f"Precision: {total_precision / count:.4f}, Recall: {total_recall / count:.4f}, F1: {total_f1 / count:.4f}")


if __name__ == "__main__":
    test_model()
