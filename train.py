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
from metrics import compute_iou , iou , class_acc
from neptune_key import NEPTUNE_API_TOKEN

def save_scores(scores, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"val_scores_epoch_{epoch}.json")
    scores_serializable = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in scores.items()}
    with open(path, "a") as f:
        json.dump(scores_serializable, f, indent=4)

def interleaving(model , mode:str):
    '''
    mode = 'seg' or 'det
    function for freezing the head that is the opposite of the mode
    eg : mode = 'seg' --> freeze the detection head
    '''
    if mode not in {'seg', 'det'}:
        raise ValueError("mode must be either 'seg' or 'det'")
    
    requires_grad_map = {
        'seg': {'seg_head': True, 'det_head': False},
        'det': {'seg_head': False, 'det_head': True}
    }
    
    for head_name, requires_grad in requires_grad_map[mode].items():
        for p in getattr(model, head_name).parameters():
            p.requires_grad = requires_grad
    model.train()
    return model


def val_one_epoch(model , epoch , args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # for p in model.parameters():
    #     p.requires_grad = True
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
    mask_iou = 0
    bbox_iou = 0
    cls_acc = 0

    for seg_batch in tqdm(seg_loader, desc=f"Segmentation Validation"): 
        
        seg_img , mask = seg_batch
        seg_img , mask = seg_img.to(device) , mask.to(device)
        
        with torch.no_grad():
            outputs_seg = model(seg_img)
            pred_mask = torch.sigmoid(outputs_seg['masks'])
            seg_iou = compute_iou(pred_mask , mask)
            # print(f"Sample mask IoU: {seg_iou.item()}")
            mask_iou+=seg_iou
            
        
    for det_batch in tqdm(det_loader, desc=f"Detection Validation"): 
        
        det_img , target = det_batch
        det_img = det_img.to(device)
        bbox , labels , obj = target["boxes"] , target["labels"] , target["obj"]
        gt_box , labels , obj = bbox.to(device) , labels.to(device) , obj.to(device)
        gt_box = torch.sigmoid(gt_box)
        with torch.no_grad():
            outputs_det = model(det_img)
        pred_box , pred_label , pred_score = torch.sigmoid(outputs_det['bbox']), torch.sigmoid(outputs_det['class_score']),  torch.sigmoid(outputs_det['obj_score'])
        obj_thresh = 0.5
        keep_mask = (pred_score > obj_thresh)

        # Convert bbox from [B, 4, H, W] to [B, H, W, 4]
        pred_bbox = pred_box.permute(0, 2, 3, 1)
        gt_box = gt_box.permute(0, 2, 3, 1)

        # Flatten everything to [B*H*W, 4] and [B*H*W]
        pred_bbox = pred_bbox.reshape(-1, 4)
        gt_box = gt_box.reshape(-1, 4)
        keep_mask = keep_mask.reshape(-1)
        # print("pred_bbox" , pred_bbox)
        # print("gt_box" , gt_box)
        # Apply the mask
        pred_xyxy = pred_bbox[keep_mask]
        gt_xyxy = gt_box[keep_mask]
        if pred_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
            det_iou = torch.tensor(0.0, device=device)
        else:
            det_iou = iou(pred_xyxy, gt_xyxy)
            bbox_iou += det_iou.mean()

     
        cls_score = class_acc(pred_label , labels , obj)
        cls_acc+=cls_score
    
    
    scores = {
        'seg_quality' : mask_iou / len(seg_loader) , 
        'det_quality' :  bbox_iou / len(det_loader) , 
        'class_quality' : cls_acc / len(det_loader)
    }
    save_scores(scores , args.out_dir , epoch )
               
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
    
    seg_dataset = Subset(seg_dataset_full , range(len(det_dataset)))
    
    seg_loader = DataLoader(dataset = seg_dataset ,
                                                    batch_size = args.train_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers , 
                                                    pin_memory=True, 
                                                    persistent_workers=True)
    
    det_loader = DataLoader(dataset = det_dataset ,
                                                    batch_size = args.train_batch,
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
    model.to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr , )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=args.lr, 
#     steps_per_epoch=len(seg_loader), 
#     epochs=args.epochs
# )

    seg_criterion = torch.nn.BCEWithLogitsLoss()
    det_criterion = torch.nn.SmoothL1Loss()
    class_criterion = torch.nn.BCEWithLogitsLoss()
    obj_criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    saving_loss = float('inf')
    
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
            seg_img , mask = seg_img.to(device) , mask.to(device)
            model = interleaving(model , mode = 'seg')
            with torch.amp.autocast(device_type='cuda'):
                output = model(seg_img)
                pred_mask = torch.sigmoid(output['masks'])
                loss_seg = seg_criterion(pred_mask , mask)
            if run:
                run["segmentation_loss/batch"].append(loss_seg)
            #detection   
            det_img , target = det_batch
            det_img = det_img.to(device)
            model = interleaving(model , mode = 'det')
            bbox , labels , obj = target["boxes"] , target["labels"] , target["obj"]
            bbox , labels , obj = bbox.to(device) , labels.to(device) , obj.to(device)
            with torch.amp.autocast(device_type='cuda'):
                output = model(det_img)
                pred_bbox , pred_score, pred_labels = output['bbox'] , output['obj_score'] , output['class_score']
                pred_bbox = torch.sigmoid(output['bbox'])
                loss_det = det_criterion(pred_bbox , bbox)
                loss_class = class_criterion(pred_labels , labels) 
                loss_obj = obj_criterion(pred_score , obj)
            total_loss = 2.0 * loss_seg + 2.0 * loss_det + 0.5 * loss_class + 0.5 * loss_obj

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
                    "optimizer_state": optimizer.state_dict(),
                    }
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
    parser.add_argument('--train_batch' , type=int , required=True , help="Training Batch Size")
    parser.add_argument("--val_batch" , type=int , required=True , help="Validation Batch Size")
    parser.add_argument('--num_workers' , type=int , default=4 , help = "number of workers for data loading")
    parser.add_argument("--lr_scheduler", type=str , default = 'exponential_decay' , help = "lr schedulers implemented : exponential_decay & ")
    parser.add_argument('--epochs' , type=int , required=True , help="number of training epochs")
    parser.add_argument('--out_dir' , type=str , required=True , help="directory to save weight file")
    parser.add_argument('--neptune' , type=bool , default=False , help='set to True for neptune logging')
    
    args = parser.parse_args()
    train(args)
    # checkpoint = torch.load("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/results/checkpoint_epoch_1.pt")
    # model = SegDet(ckpt_path=None).to('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(checkpoint["model_state"])
    # model.eval()
    # scores = val_one_epoch(model , args)
    # print(scores)