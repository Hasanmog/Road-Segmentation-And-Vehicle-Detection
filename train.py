import argparse
import torch
import os
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from data.dataloader import RoadSegDataset , VehicleDetDataset 
from model.model import SegDet 

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
    
def train(args):
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
                                                    num_workers = args.num_workers)
    
    det_loader = DataLoader(dataset = det_dataset ,
                                                    batch_size = args.train_batch,
                                                    shuffle = True , 
                                                    num_workers = args.num_workers)
    print(f"There is {len(seg_dataset)} segmentation training samples")
    print(f"There is {len(det_dataset)} detection training samples")
    
    if args.freeze_backbone:
        assert args.ckpt_path != None , "To freeze backbone , we need weight file path in the '--ckpt_path' argument"
        
    model = SegDet(img_size = args.img_size,
                                small_patch_size = args.small_patch,
                                large_patch_size = args.large_patch,
                                ckpt_path = args.ckpt_path ,
                                backbone_freeze = args.freeze_backbone)
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    det_criterion = torch.nn.SmoothL1Loss()
    class_criterion = torch.nn.BCEWithLogitsLoss()
    obj_criterion = torch.nn.BCEWithLogitsLoss()

    
    
    for epoch in range(args.epochs):
       for (seg_batch , det_batch) in tqdm(zip(seg_loader , det_loader), desc=f"Epoch {epoch+1}"):
           #segmentation
            optimizer.zero_grad()
            seg_img , mask = seg_batch
            seg_img , mask = seg_img.to(device) , mask.to(device)
            model = interleaving(model , mode = 'seg')
            output = model(seg_img)
            pred_mask = output['masks']
            loss_seg = seg_criterion(pred_mask , mask)
            
            #detection   
            det_img , target = det_batch
            det_img = det_img.to(device)
            model = interleaving(model , mode = 'det')
            bbox , labels , obj = target["boxes"] , target["labels"] , target["obj"]
            bbox , labels , obj = bbox.to(device) , labels.to(device) , obj.to(device)
            output = model(det_img)
            pred_bbox , pred_score, pred_labels = output['bbox'] , output['obj_score'] , output['class_score']
            # print("pred labels" , pred_labels.shape)
            # print("gt labels" , labels.shape)
            loss_det = det_criterion(pred_bbox , bbox)
            loss_class = class_criterion(pred_labels , labels) 
            loss_obj = obj_criterion(pred_score , obj)
            total_loss = loss_seg + loss_det + loss_class + loss_obj 
            total_loss.backward()
            optimizer.step()
       print(f"[Epoch {epoch}] Seg Loss: {loss_seg.item():.4f} | Det Loss: {loss_det.item():.4f} | Obj: {loss_obj.item():.4f} | Cls: {loss_class.item():.4f}")
       checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"))

               
               
               
               
       
       
       
    
        
    
    
    
                            
    
    
    

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
    parser.add_argument('--lr' , type=float , default=1e-4 , help="Learning Rate")
    parser.add_argument('--train_batch' , type=int , required=True , help="Training Batch Size")
    parser.add_argument("--val_batch" , type=int , required=True , help="Validation Batch Size")
    parser.add_argument('--num_workers' , type=int , default=4 , help = "number of workers for data loading")
    parser.add_argument('--epochs' , type=int , required=True , help="number of training epochs")
    parser.add_argument('--out_dir' , type=str , required=True , help="directory to save weight file")
    
    args = parser.parse_args()
    train(args)