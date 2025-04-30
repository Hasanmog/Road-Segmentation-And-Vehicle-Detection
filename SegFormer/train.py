import os
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from torch.cuda.amp import autocast, GradScaler
from data.dataloader import RoadSegDataset
import torchvision.transforms as T

IMG_SIZE = 512
NUM_CLASSES = 2
BATCH_SIZE = 4
EPOCHS = 10
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_dir = "checkpoints_2"
os.makedirs(ckpt_dir, exist_ok=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

weights = torch.tensor([0.1, 0.9]).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

root = "/home/hasanmog/datasets/dataset_zoom_original/dataset_zoom_original/"

augmentations = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(15),
])

train_dataset = RoadSegDataset(root, IMG_SIZE, "train", augment=True, augmentations=augmentations)
val_dataset = RoadSegDataset(root, IMG_SIZE, "val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True , num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False , num_workers = 2)

def validate(model, dataloader):
    model.eval()
    total_iou = 0
    total_acc = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)

            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            iou = jaccard_score(labels_np, preds_np, average="macro", zero_division=1)
            acc = (preds_np == labels_np).sum() / len(labels_np)

            total_iou += iou
            total_acc += acc
            count += 1

    return total_iou / count, total_acc / count

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(pixel_values=pixel_values)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

    miou, acc = validate(model, val_loader)
    print(f"[Epoch {epoch+1}] Val mIoU: {miou:.4f}, Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"segformer_epoch{epoch+1}.pth"))
