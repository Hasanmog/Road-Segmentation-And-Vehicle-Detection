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

# validation
def val_one_epoch(model , epoch , args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    seg_val = RoadSegDataset(dataset_dir=args.seg_data_dir , 
                                                        mode = 'val' , 
                                                        img_size = args.img_size)
    
    det_val = VehicleDetDataset(dataset_dir=args.det_data_dir , 
                                                         mode = "val" , 
                                                         img_size = args.img_size)
    
    seg_loader = DataLoader(dataset = seg_val ,
                                                    batch_size = args.val_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers,
                                                    pin_memory=True, 
                                                    persistent_workers=True)
    
    det_loader = DataLoader(dataset = det_val ,
                                                    batch_size = args.val_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers , 
                                                    pin_memory=True, 
                                                    persistent_workers=True)
    print(f"There are {len(seg_val)} segmentation validation samples")
    print(f"There are {len(det_val)} detection validation samples")
    
    seg_acc = 0
    bbox_acc = 0
    cls_acc = 0

    for seg_batch in tqdm(seg_loader, desc=f"Segmentation Validation"): 
        
        seg_img , mask = seg_batch
        seg_img , gt_mask = seg_img.to(device) , mask.to(device)
        
        with torch.no_grad():
            outputs_seg = model(seg_img)
            
            pred_mask = torch.sigmoid(outputs_seg['masks'])
            
            seg_iou = mask_iou(pred_mask , gt_mask)
            
            seg_acc += seg_iou.item()
            
    for det_batch in tqdm(det_loader, desc=f"Detection Validation"): 
        
        det_img , target = det_batch
        det_img = det_img.to(device)
        gt_box = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)
        gt_obj = target["obj"].to(device)
        
        with torch.no_grad():
            outputs_det = model(det_img)
            pred_box = outputs_det['bbox']       
            pred_label = outputs_det['class_score']  
            pred_score = torch.sigmoid(outputs_det['obj_score'])
        valid_iou_batches = 0
        for b in range(det_img.size(0)):
            pred_bbox_b = pred_box[b]       # [4, H, W]
            pred_score_b = pred_score[b]    # [H, W]
            pred_label_b = pred_label[b]
            gt_bbox_b = gt_box[b]
            gt_obj_b = gt_obj[b]
            gt_label_b = gt_labels[b]

            pred_mask = pred_score_b > 0.1
            gt_mask = gt_obj_b == 1

            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                continue

            pred_boxes = pred_bbox_b[:, pred_mask].T  # [N_pred, 4]
            gt_boxes = gt_bbox_b[:, gt_mask].T        # [N_gt, 4]

            if pred_boxes.shape[0] == gt_boxes.shape[0]:
                det_iou = box_iou(pred_boxes, gt_boxes).mean().item()
                bbox_acc += det_iou
                valid_iou_batches += 1

            cls_score = class_acc(pred_label_b, gt_label_b, gt_obj_b)
            cls_acc += cls_score

    scores = {
        'seg_quality': seg_acc / len(seg_loader),
        'det_quality': bbox_acc / max(1, valid_iou_batches),
        'class_quality': cls_acc / len(det_loader)
    }

    save_scores(scores, args.out_dir, epoch)

#Full Training                
def train(args):
    
    run = None
    if args.neptune:
        run = neptune.init_run(
        project="ham82/RoadSeg-VehicleDet",
        api_token=NEPTUNE_API_TOKEN,
        )  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    seg_dataset_full = RoadSegDataset(dataset_dir=args.seg_data_dir , 
                                                        mode = 'train' , 
                                                        img_size = args.img_size)
    
    det_dataset = VehicleDetDataset(dataset_dir=args.det_data_dir , 
                                                         mode = "train" , 
                                                         img_size = args.img_size)
    
    seg_dataset = Subset(seg_dataset_full , range(len(det_dataset))) # to match the size of detection set
    
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
    
    # Trainable param vs Total number of param
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable_params:,} out of {total_params:,} parameters "
        f"({100 * trainable_params / total_params:.2f}%)")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr , weight_decay = args.weight_decay)
    scheduler = pick_scheduler(optimizer , scheduler = args.lr_scheduler)
    
    seg_criterion = torch.nn.BCEWithLogitsLoss() # added to the dice loss 
    det_criterion = torch.nn.SmoothL1Loss()
    class_criterion = torch.nn.BCEWithLogitsLoss()
    
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
                pred_bbox , pred_score, pred_labels = output['bbox'] , output['obj_score'] , output['class_score']
                
                mask = gt_obj== 1
                mask = mask.unsqueeze(1).expand_as(pred_bbox)
                pred_bbox = pred_bbox[mask].view(-1, 4)
                gt_bbox = gt_bbox[mask].view(-1, 4)
                if pred_bbox.numel() > 0 and gt_bbox.numel() > 0:
                    loss_det = det_criterion(pred_bbox, gt_bbox)
                else:
                    loss_det = torch.tensor(0.0, device=pred_bbox.device)

                mask = gt_obj == 1

                if mask.sum() > 0:
                    loss_class = focal_loss(pred_labels[mask], gt_labels[mask], gamma=2.0, alpha=0.25)
                else:
                    loss_class = torch.tensor(0.0, device=pred_labels.device)
    
                loss_obj = focal_loss(torch.sigmoid(pred_score) , gt_obj)
                
            total_loss = 2.0 * loss_seg + 1.5 * loss_det + 0.5 * loss_class + 0.5 * loss_obj

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
                    
        val_one_epoch(model = model , epoch = epoch , args=args)
        if  total_loss < saving_loss:
            saving_loss = total_loss      
            checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),}
            
            torch.save(checkpoint, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    if run:
        run.stop()

               
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for segmentation and detection")
    
    # Dataset-related args
    parser.add_argument('--seg_data_dir' , type =str , required=True , help="Directory where the whole segmentation dataset is stored")
    parser.add_argument('--det_data_dir' , type =str , required=True , help="Directory where the whole detection dataset is stored")
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
    # checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_5.pt")
    # model = SegDet(ckpt_path=None).to('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    # scores = val_one_epoch(model , epoch = 5 , args = args)
    # print(scores)