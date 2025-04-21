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
    def __init__(self, dataset_dir: str, img_size: int, mode: str, grid_size: int = 64):
        self.dataset = dataset_dir
        if mode not in ["train", "val", "test"]:
            raise NameError("Mode must be one of: 'train', 'val', 'test'")
        
        self.dir = os.path.join(dataset_dir, mode)
        self.img_size = img_size
        self.grid_size = grid_size
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
        # print("raw_anno" , anns)
        obj_target = torch.zeros((self.grid_size, self.grid_size))
        cls_target = torch.zeros((self.grid_size, self.grid_size))
        bbox_target = torch.zeros((4, self.grid_size, self.grid_size))

        id_map = {1: 0, 2: 1}
        stride = self.img_size / self.grid_size

        for ann in anns:
            if ann["category_id"] not in id_map:
                continue
            label = id_map[ann["category_id"]]
            x1, y1, x2, y2 = self._resize_bbox_to_xyxy(ann["bbox"], original_w, original_h)
            # print(f"original_w: {original_w}, original_h: {original_h}")
            # print(f"image_path: {image_path}")
            # for ann in anns:
            #     print("bbox:", ann["bbox"])
            #     x1, y1, x2, y2 = self._resize_bbox_to_xyxy(ann["bbox"], original_w, original_h)
            #     # print(f"Resized xyxy: {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}")


            gi = int((x1 + x2) / 2 / stride)
            gj = int((y1 + y2) / 2 / stride)
            if gi >= self.grid_size or gj >= self.grid_size:
                continue
            obj_target[gj, gi] = 1
            cls_target[gj, gi] = label
            bbox_target[0, gj, gi] = x1 / self.img_size
            bbox_target[1, gj, gi] = y1 / self.img_size
            bbox_target[2, gj, gi] = x2 / self.img_size
            bbox_target[3, gj, gi] = y2 / self.img_size
        # print("Final obj_target max:", obj_target.max())
        # print("Final labels unique:", cls_target.unique())
        # print("Final bbox_target sum:", bbox_target.sum())
        # print(bbox_target.shape)
        target = {
            "obj": obj_target,
            "labels": cls_target,
            "boxes": bbox_target,
            "image_id": image_id
        }
        return image, target
    
    
               
if __name__ == "__main__":
    det_dataset = VehicleDetDataset(dataset_dir="/home/hasanmog/datasets/vedai" , 
                                                         mode = "train" , 
                                                         img_size = 512)           
    
    sample = det_dataset[0]
             