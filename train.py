import argparse
import torch
import os
import json
import neptune
from tqdm import tqdm
from torch.utils.data import Subset
from torch.amp import autocast,GradScaler
from torch.utils.data import DataLoader
from data.dataloader import RoadSegDataset , VehicleDetDataset
from data.postprocess import postprocess_bbox 
from model.model import SegDet 
from metrics import mask_iou , box_iou , class_acc , dice_loss , focal_loss
from neptune_key import NEPTUNE_API_TOKEN
from train_utils import save_scores , interleaving , pick_scheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert



def val_one_epoch(model , epoch , args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    grid_size = args.img_size // args.small_patch

    seg_val = RoadSegDataset(dataset_dir=args.seg_data_dir, mode='val', img_size=args.img_size)
    det_val = VehicleDetDataset(dataset_dir=args.det_data_dir, mode="val", img_size=args.img_size, grid_size=grid_size)

    seg_loader = DataLoader(seg_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    det_loader = DataLoader(det_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    print(f"There are {len(seg_val)} segmentation validation samples")
    print(f"There are {len(det_val)} detection validation samples")

    seg_acc = 0
    metric = MeanAveragePrecision(iou_type="bbox")

    for seg_img, mask in tqdm(seg_loader, desc="Segmentation Validation"):
        seg_img, gt_mask = seg_img.to(device), mask.to(device)
        with torch.no_grad():
            outputs_seg = model(seg_img)
            pred_mask = torch.sigmoid(outputs_seg['masks'])
            seg_iou = mask_iou(pred_mask, gt_mask)
            seg_acc += seg_iou.item()

    for det_img, target in tqdm(det_loader, desc="Detection Validation"):
        det_img = det_img.to(device)
        with torch.no_grad():
            outputs_det = model(det_img)
            pred_box = outputs_det['bbox']
            pred_label = outputs_det['class_score']
            pred_score = torch.sigmoid(outputs_det['obj_score'])

        batch_preds = []
        batch_targets = []

        for b in range(det_img.size(0)):
            pred_bbox_b = pred_box[b].view(4, -1)
            pred_score_b = pred_score[b].flatten()
            pred_label_b = pred_label[b].permute(1, 2, 0).reshape(-1, pred_label.shape[1])
            mask_flat = pred_score_b > 0.5

            if mask_flat.sum() == 0:
                continue

            pred_boxes = pred_bbox_b[:, mask_flat].T
            pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            scores = pred_score_b[mask_flat]
            labels = torch.argmax(pred_label_b[mask_flat], dim=1)

            batch_preds.append({
                "boxes": pred_boxes.detach().cpu(),
                "scores": scores.detach().cpu(),
                "labels":  labels.detach().cpu().to(torch.int64)
            })

            gt_bbox_b = target['boxes'][b].view(4, -1)
            gt_obj_b = target['obj'][b].flatten()
            gt_label_b = target['labels'][b].flatten()
            gt_mask = gt_obj_b == 1

            gt_boxes = gt_bbox_b[:, gt_mask].T
            gt_boxes = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            gt_labels = gt_label_b[gt_mask]

            batch_targets.append({
                "boxes": gt_boxes.detach().cpu(),
                "labels": gt_labels.detach().cpu().to(torch.int64)
            })

        metric.update(batch_preds, batch_targets)

    map_results = metric.compute()

    scores = {
        'seg_quality': seg_acc / len(seg_loader),
        'mAP@0.5': map_results['map_50'].item(),
        'mAP@[0.5:0.95]': map_results['map'].item()
    }

    save_scores(scores, args.out_dir, epoch)
    return scores



#Full Training                
def train(args):
    
    run = None
    if args.neptune:
        run = neptune.init_run(
        project="ham82/RoadSeg-VehicleDet",
        api_token=NEPTUNE_API_TOKEN,
        )  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    seg_dataset= RoadSegDataset(dataset_dir=args.seg_data_dir , 
                                                        mode = 'train' , 
                                                        img_size = args.img_size,
                                                        target_len=args.dataset_len)
    grid_size = args.img_size // args.small_patch
    det_dataset = VehicleDetDataset(dataset_dir=args.det_data_dir , 
                                                         mode = "train" , 
                                                         img_size = args.img_size , 
                                                         target_len= len(seg_dataset) , 
                                                         grid_size=grid_size)
    
    
    seg_loader = DataLoader(dataset = seg_dataset ,
                                                    batch_size = args.seg_train_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers , 
                                                    pin_memory=True, 
                                                    persistent_workers=True)
    
    det_loader = DataLoader(dataset = det_dataset ,
                                                    batch_size = args.det_train_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers , 
                                                    pin_memory=True, 
                                                    persistent_workers=True)
    print(f"There are {len(seg_dataset)} segmentation training samples")
    print(f"There are {len(det_dataset)} detection training samples")
    
    if args.freeze_backbone:
        assert args.ckpt_path != None , "To freeze backbone , we need weight file path in the '--ckpt_path' argument"
          
    model = SegDet(img_size = args.img_size,
                                small_patch_size = args.small_patch,
                                large_patch_size = args.large_patch,
                                ckpt_path = args.ckpt_path ,
                                backbone_freeze = args.freeze_backbone)
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        
    model.to(device)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable_params:,} out of {total_params:,} parameters "
        f"({100 * trainable_params / total_params:.2f}%)")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr , weight_decay = args.weight_decay)
    scheduler = pick_scheduler(optimizer , scheduler = args.lr_scheduler)
    
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    det_criterion = torch.nn.SmoothL1Loss()
    class_criterion = torch.nn.CrossEntropyLoss()
    object_criterion = torch.nn.BCEWithLogitsLoss(
        
    )
    scaler = GradScaler()
    
    saving_loss = float('inf')
    model.train()
    for epoch in range(args.epochs):
        seg_loss_total = 0
        det_loss_total = 0
        obj_loss_total = 0
        cls_loss_total = 0
        total_batches = 0
        for (seg_batch , det_batch) in tqdm(zip(seg_loader , det_loader), desc=f"Epoch {epoch+1}"):
            
           #segmentation
            total_batches += 1
            optimizer.zero_grad()
            seg_img , mask = seg_batch
            seg_img , gt_mask = seg_img.to(device) , mask.to(device)
            model = interleaving(model , mode = 'seg')
            with torch.amp.autocast(device_type='cuda'):
                output = model(seg_img)
                pred_mask = output['masks']
                loss_seg = seg_criterion(pred_mask, gt_mask) + dice_loss(torch.sigmoid(pred_mask), gt_mask)
            if run:
                run["segmentation_loss/batch"].append(loss_seg)
                
            #detection   
            det_img , target = det_batch
            det_img = det_img.to(device)
            model = interleaving(model , mode = 'det')
            bbox , labels , obj = target["boxes"] , target["labels"] , target["obj"]
            gt_bbox , gt_labels , gt_obj = bbox.to(device) , labels.to(device) , obj.to(device)
            with torch.amp.autocast(device_type='cuda'):
                output = model(det_img)
                pred_bbox = output['bbox']       # [B, 4, H, W]
                pred_score = output['obj_score'] # [B, 1, H, W]
                pred_labels = output['class_score'] # [B, C, H, W]
                mask = gt_obj == 1                       # [B, H, W]
                mask_bbox = mask.unsqueeze(1).expand_as(pred_bbox)  # [B, 4, H, W]
                
                pred_bbox = pred_bbox[mask_bbox].view(-1, 4)
                gt_bbox = gt_bbox[mask_bbox].view(-1, 4)

                if pred_bbox.numel() > 0 and gt_bbox.numel() > 0:
                    loss_det = det_criterion(pred_bbox, gt_bbox)
                else:
                    loss_det = torch.tensor(0.0, device=pred_bbox.device)


                mask = gt_obj == 1

               # Inside training loop:
                loss_class = 0.0
                for b in range(pred_labels.size(0)):
                    mask = gt_obj[b] == 1
                    if mask.sum() == 0:
                        continue

                    # pred_labels[b]: [2, H, W] → [H, W, 2]
                    pred_labels_b = pred_labels[b].permute(1, 2, 0)  # [H, W, 2]
                    pred_pos = pred_labels_b[mask]                  # [N, 2]

                    gt_pos = gt_labels[b][mask].long()              # [N]
                    loss_class += class_criterion(pred_pos, gt_pos)

                loss_class = loss_class / pred_labels.size(0)




                loss_obj = object_criterion(pred_score, gt_obj.unsqueeze(1).float())
                
            total_loss = 1.5 * loss_seg + 1.5 * loss_det + 1.0 * loss_class + 0.5 * loss_obj
            if run:
                    run["detection_loss/batch"].append(loss_det)
                    run["classification_loss/batch"].append(loss_class)
                    run["objectscore_loss/batch"].append(loss_obj)
                    run["total_loss/batch"].append(total_loss)
                    
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if run:
                current_lr = scheduler.get_last_lr()[0]
                run["learning_rate"].append(current_lr)
                
            seg_loss_total += loss_seg.item()
            det_loss_total += loss_det.item()
            obj_loss_total += loss_obj.item()
            cls_loss_total += loss_class.item()
            total_loss = seg_loss_total + det_loss_total + obj_loss_total + cls_loss_total
            
        scheduler.step()
        
        print(f"[Epoch {epoch+1}] Seg Loss: {seg_loss_total / total_batches:.4f} | "
          f"Det Loss: {det_loss_total / total_batches:.4f} | "
          f"Obj: {obj_loss_total / total_batches:.4f} | "
          f"Cls: {cls_loss_total / total_batches:.4f}")
        
        if run:
                    run["segmentation_loss/epoch"].append(seg_loss_total / total_batches)
                    run["detection_loss/epoch"].append(det_loss_total / total_batches)
                    run["classification_loss/epoch"].append(cls_loss_total / total_batches)
                    run["objectscore_loss/epoch"].append(obj_loss_total / total_batches)
                    run["total_loss/epoch"].append(total_loss / total_batches)
                    
        if  total_loss < saving_loss:
            saving_loss = total_loss      
            checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),}
            
            torch.save(checkpoint, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            
        val_one_epoch(model = model , epoch = epoch , args=args)
    if run:
        run.stop()

               
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for segmentation and detection")
    
    # Dataset-related args
    parser.add_argument('--seg_data_dir' , type =str , required=True , help="Directory where the whole segmentation dataset is stored")
    parser.add_argument('--det_data_dir' , type =str , required=True , help="Directory where the whole detection dataset is stored")
    parser.add_argument('--dataset_len' , type = int , default = 3000 , help="if dataset is smaller , augmentation is applied to reach this size")
    parser.add_argument('--img_size' , type=int , default=512 , help="Size of the image")
    
    #Model-related args
    parser.add_argument('--freeze_backbone' , type=bool , default=True , help="if set , backbone(image encoder) will be freezed")
    parser.add_argument('--ckpt_path' , type=str , default=None , help="path of the image encoder checkpoint(SAM)")
    parser.add_argument('--small_patch' , type=int , default=8 , help = "Size of the small(local) patch size")
    parser.add_argument('--large_patch' , type=int , default=16 , help = "Size of the large(global) patch size")
    
    #Training-related args
    parser.add_argument('--lr' , type=float , default=1e-5 , help="Learning Rate")
    parser.add_argument('--weight_decay' , type=float , default=5e-4 , help="Training Weight Decay")
    parser.add_argument('--seg_train_batch' , type=int , required=True , help="Segmentation Training Batch Size")
    parser.add_argument('--det_train_batch' , type=int , required=True , help="Detection Training Batch Size")
    parser.add_argument("--val_batch" , type=int , required=True , help="Validation Batch Size")
    parser.add_argument('--num_workers' , type=int , default=4 , help = "Number Of Workers For Data Loading")
    parser.add_argument("--lr_scheduler", type=str , default = 'exponential' , help = "lr schedulers implemented : exponential_decay & ")
    parser.add_argument('--epochs' , type=int , required=True , help="number of training epochs")
    parser.add_argument('--out_dir' , type=str , required=True , help="directory to save weight file")
    parser.add_argument('--neptune' , type=bool , default=False , help='set to True for neptune logging')
    parser.add_argument('--resume_checkpoint' , type=str , default=None , help="Checkpoint to resume training from")

    
    args = parser.parse_args()
    train(args)
    # checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_1.pt")
    # model = SegDet(ckpt_path=None).to('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    # scores = val_one_epoch(model , epoch = 5 , args = args)
    # print(scores)