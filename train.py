import torch 
import os
import torch.nn as nn
import argparse
import neptune
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast,GradScaler
from neptune_key import NEPTUNE_API_TOKEN
from model.model import SegDet 
from data.dataloader import RoadSegDataset , VehicleDetDataset
from criterion import seg_criterion , det_criterion
from train_utils import interleaving , lr_scheduler
from val import validate_one_epoch

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
    
    det_dataset = VehicleDetDataset(dataset_dir=args.det_data_dir , 
                                                         mode = "train" , 
                                                         img_size = args.img_size , 
                                                         target_len= len(seg_dataset) , 
                                                         grid_size = 8)

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
    
    seg_val = RoadSegDataset(dataset_dir=args.seg_data_dir, mode='val', img_size=args.img_size)
    det_val = VehicleDetDataset(dataset_dir=args.det_data_dir, mode="val", img_size=args.img_size, grid_size=8)

    seg_val_loader = DataLoader(seg_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    det_val_loader = DataLoader(det_val, batch_size=args.val_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"There are {len(seg_dataset)} segmentation training samples")
    print(f"There are {len(det_dataset)} detection training samples")
    print(f"There are {len(seg_val)} segmentation validation samples")
    print(f"There are {len(det_val)} detection validation samples")
    
    model = SegDet(img_size = args.img_size,
                                small_patch_size = args.small_patch,
                                large_patch_size = args.large_patch,
                                backbone = args.backbone,
                                sam_ckpt_path=args.sam_ckpt , 
                                swin_det_path= args.swin_det_ckpt ,
                                swin_seg_path= args.swin_seg_ckpt, 
                                backbone_freeze = args.freeze_backbone)
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable_params:,} out of {total_params:,} parameters "
        f"({100 * trainable_params / total_params:.2f}%)")
    
    
    optimizer = torch.optim.AdamW(
                                                            filter(lambda p: p.requires_grad, model.parameters()), 
                                                            lr=args.lr,
                                                            weight_decay=args.weight_decay,
                                                            betas=(args.beta1, args.beta2))
    scaler = GradScaler()
    
    saving_loss = float('inf')
    model.train()
    scheduler = lr_scheduler(optimizer , scheduler = args.lr_scheduler)
    for epoch in tqdm(range(args.epochs)):
        
        seg_loss_total = 0
        det_loss_total = 0
        total = 0
        i = 0
        for (seg_batch , det_batch) in tqdm(zip(seg_loader , det_loader)):
            optimizer.zero_grad()
            seg_img , mask = seg_batch
            seg_img , gt_mask = seg_img.to(device) , mask.to(device)
            
            model = interleaving(model , mode = 'seg')
            
            with torch.amp.autocast(device_type=device):
                output = model(seg_img)

                class_logits = output["mask_logits"]  # [B, 100, 1]
                masks = output["masks"].squeeze(1)    # [B, 100, 512, 512] after fix

                best_queries = class_logits.squeeze(-1).sigmoid().max(dim=1)[1]  # [B]
                
                batch_size = masks.size(0)
                selected_masks = torch.stack([masks[b, best_queries[b]] for b in range(batch_size)], dim=0)  # [B, 512, 512]

                selected_masks = selected_masks.unsqueeze(1)  # [B, 1, 512, 512]
                seg_loss = seg_criterion(selected_masks, gt_mask)
                seg_loss_total += seg_loss

            if run:
                run["segmentation_loss/batch"].append(seg_loss.item())

                
            det_img , target = det_batch
            det_img = det_img.to(device)
            
            model = interleaving(model , mode = 'det')
            cls , bbox , center = target["cls"] , target["bbox"] , target["centerness"]
            gt_box , gt_label , gt_center = bbox.to(device) , cls.to(device) , center.to(device)
            
            with torch.amp.autocast(device_type=device):
                output = model(det_img)
                pred_box = output["bbox"]
                pred_label = output["cls_logits"]
                pred_center = output["centerness"]
                
                pred = [pred_label , pred_box , pred_center]
                gt = [gt_label , gt_box , gt_center]
                det_loss , losses = det_criterion(pred , gt , weight=[0.5 , 1 , 0.5])
                det_loss_total+=det_loss
            if run:
                run["detection_total_loss/batch"].append(det_loss.item())
                run["classification_loss/batch"].append(losses[0].item())
                run["detection_loss/batch"].append(losses[1].item())
                run["center_loss/batch"].append(losses[2].item())
                current_lr = scheduler.get_last_lr()[0]
                run["learning_rate"].append(current_lr)
            total_loss = 2 * seg_loss + det_loss
            total+=total_loss
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← this line applies gradient clipping

            scaler.step(optimizer)
            scaler.update()
            if args.lr_scheduler in ["onecycle", "cosine", "cosine_warmup"]:
                scheduler.step()
                
            if run:
                run["total_loss/batch"].append(total_loss.item())
                
        if args.lr_scheduler not in ["onecycle", "cosine"]:
            scheduler.step()
    
        total_seg = seg_loss_total / len(seg_loader)
        total_det = det_loss / len(det_loader)
        total = total / args.dataset_len
        run["segmentation_loss/epoch"].append(total_seg.item())
        run["detection_loss/epoch"].append(total_det.item())
        run["Total_loss/epoch"].append(total.item())
        
        if total < saving_loss:
            saving_loss = total     
            checkpoint = {
                                    "epoch": epoch,
                                    "model_state": model.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                    "scaler_state": scaler.state_dict(),      
                                    "scheduler_state": scheduler.state_dict(),   
                                    "saving_loss": total                   
                                }

            os.makedirs(args.out_dir, exist_ok=True)    
            torch.save(checkpoint, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        print("#######################VALIDATION#######################")
        validate_one_epoch(model , seg_val_loader , det_val_loader , device , run = run , args = args) 
            
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
    parser.add_argument('--backbone' , type=str , default="SWIN" , help="SWIN OR SAM")
    parser.add_argument('--freeze_backbone' , action='store_true' , default=True , help="if set , backbone(image encoder) will be freezed")
    parser.add_argument('--sam_ckpt' , type=str , default=None , help="path of the image encoder checkpoint(SAM)")
    parser.add_argument('--swin_det_ckpt' , type=str , default=None , help="path to the swin detection checkpoint")
    parser.add_argument("--swin_seg_ckpt" , type=str , default=None , help="path to the swin segmentation path")
    parser.add_argument('--small_patch' , type=int , default=8 , help = "Size of the small(local) patch size")
    parser.add_argument('--large_patch' , type=int , default=16 , help = "Size of the large(global) patch size")
    
    #Training-related args
    parser.add_argument('--lr' , type=float , default=1e-5 , help="Learning Rate")
    parser.add_argument('--beta1' , type=float , default=0.9)
    parser.add_argument('--beta2' , type=float , default=0.999)
    parser.add_argument('--weight_decay' , type=float , default=5e-4 , help="Training Weight Decay")
    parser.add_argument('--seg_train_batch' , type=int , required=True , help="Segmentation Training Batch Size")
    parser.add_argument('--det_train_batch' , type=int , required=True , help="Detection Training Batch Size")
    parser.add_argument("--val_batch" , type=int , required=True , help="Validation Batch Size")
    parser.add_argument('--num_workers' , type=int , default=2 , help = "Number Of Workers For Data Loading")
    parser.add_argument("--lr_scheduler", type=str , default = 'exponential' , help = "lr schedulers implemented : exponential_decay & ")
    parser.add_argument('--epochs' , type=int , required=True , help="number of training epochs")
    parser.add_argument('--out_dir' , type=str , required=True , help="directory to save weight file")
    parser.add_argument('--neptune', action='store_true', help='Enable Neptune logging')

    parser.add_argument('--resume_checkpoint' , type=str , default=None , help="Checkpoint to resume training from")

    
    args = parser.parse_args()
    
    train(args)