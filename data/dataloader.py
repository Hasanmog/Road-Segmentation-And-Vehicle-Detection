import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoadSegDataset(Dataset):
    def __init__(self , 
                     dataset_dir : str , 
                     img_size : int , 
                     mode : str ):
        
        self.dataset_dir = dataset_dir  
        
        if mode == "train" or mode == "val" or mode == "test":
            self.imgs , self.labels = os.path.join(self.dataset_dir , f"{mode}/images") , os.path.join(self.dataset_dir , f"{mode}/labels")
            
        else:
            raise NameError("Mode must be one of: 'train', 'val', 'test'")
        
        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
                                                                T.Resize((img_size, img_size)),
                                                                T.ToTensor(),                    
                                                            ])
        
        self.images = sorted([
            f for f in os.listdir(self.imgs) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.masks = sorted([
            f for f in os.listdir(self.labels) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        
    def __len__(self):
        
        assert len(self.images) == len(self.masks) , "Mismatch in the number of images and masks"
        return len(self.images)
    
    
    def __getitem__(self , idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.imgs , img_name)
        mask_path = os.path.join(self.labels , img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask    
        
class VehicleDetDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int, mode: str):
        self.dataset = dataset_dir
        if mode in ["train", "val", "test"]:
            self.dir = os.path.join(dataset_dir, mode)
        else: 
            raise NameError("Mode must be one of: 'train', 'val', 'test'")
        
        self.img_size = img_size
        self.imgs = sorted([f for f in os.listdir(self.dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.anno = [f for f in os.listdir(self.dir) if f.endswith('.json')]

        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        with open(os.path.join(self.dir, self.anno[0]), "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.imgs)
    
    def _resize_bbox_to_xyxy(self, bbox, orig_w, orig_h):
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        x, y, w, h = bbox
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + w) * scale_x
        y2 = (y + h) * scale_y
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image_path = os.path.join(self.dir, image_name)
        raw_image = Image.open(image_path).convert("RGB")
        original_w, original_h = raw_image.size
        image = self.image_transform(raw_image)

        image_id = None
        for img_info in self.annotations["images"]:
            if img_info["file_name"] == image_name:
                image_id = img_info["id"]
                break

        anns = [ann for ann in self.annotations["annotations"] if ann["image_id"] == image_id]

        boxes = []
        labels = []
        id_map = {1: 0, 2: 1}
        for ann in anns:
            if ann["category_id"] in id_map:
                xyxy_box = self._resize_bbox_to_xyxy(ann["bbox"], original_w, original_h)
                boxes.append(xyxy_box)
                labels.append(id_map[ann["category_id"]])


        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": image_id
        }

        return image, target

        
             
             