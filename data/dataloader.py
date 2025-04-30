import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import random



class RoadSegDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int, mode: str, target_len: int = None):
        self.dataset_dir = dataset_dir

        if mode not in ["train", "val", "test"]:
            raise NameError("Mode must be one of: 'train', 'val', 'test'")

        self.imgs = os.path.join(self.dataset_dir, f"{mode}/images")
        self.labels = os.path.join(self.dataset_dir, f"{mode}/labels")

        self.images = sorted([
            f for f in os.listdir(self.imgs) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.masks = sorted([
            f for f in os.listdir(self.labels) if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.true_len = len(self.images)
        self.target_len = target_len if target_len and target_len > self.true_len else self.true_len
        self.mode = mode
        self.img_size = img_size

        
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        
        self.augment = T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=15)
        ], p=0.8)

        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        true_idx = idx % self.true_len
        img_name = self.images[true_idx]
        img_path = os.path.join(self.imgs, img_name)
        mask_path = os.path.join(self.labels, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        
        image = self.base_transform(image)
        mask = self.base_transform(mask)

        if self.mode == "train" and idx >= self.true_len:
            
            seed = random.randint(0, 99999)
            torch.manual_seed(seed)
            image = self.augment(image)
            torch.manual_seed(seed)
            mask = self.augment(mask)

        
        image = self.normalize(image)

        return image, mask.float()

        

class VehicleDetDataset(Dataset):
    def __init__(self, dataset_dir, img_size, mode, grid_size=64, target_len=None):
        self.dataset = dataset_dir
        if mode not in ["train", "val", "test"]:
            raise NameError("Mode must be one of: 'train', 'val', 'test'")

        self.dir = os.path.join(dataset_dir, mode)
        self.img_size = img_size
        self.grid_size = grid_size
        self.mode = mode

        self.imgs = sorted([f for f in os.listdir(self.dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.anno = [f for f in os.listdir(self.dir) if f.endswith('.json')]

        self.true_len = len(self.imgs)
        self.target_len = target_len if target_len and target_len > self.true_len else self.true_len

        with open(os.path.join(self.dir, self.anno[0]), "r") as f:
            self.annotations = json.load(f)

        self.id_map = {1: 0, 2: 1}

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5)
            ], p=0.9),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.3))

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        true_idx = idx % self.true_len
        image_name = self.imgs[true_idx]
        image_path = os.path.join(self.dir, image_name)
        raw_image = np.array(Image.open(image_path).convert("RGB"))

        image_id = None
        for img_info in self.annotations["images"]:
            if img_info["file_name"] == image_name:
                image_id = img_info["id"]
                break

        anns = [ann for ann in self.annotations["annotations"]
                if ann["image_id"] == image_id and ann["category_id"] in self.id_map]

        bboxes = []
        category_ids = []
        for ann in anns:
            bbox = ann["bbox"]
            bboxes.append(bbox)
            category_ids.append(self.id_map[ann["category_id"]])

        if self.mode == "train" and len(bboxes) > 0:
            transformed = self.transform(image=raw_image, bboxes=bboxes, category_ids=category_ids)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["category_ids"]
        else:
            no_aug = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

            transformed = no_aug(image=raw_image, bboxes=bboxes, category_ids=category_ids)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["category_ids"]

        stride = self.img_size / self.grid_size
        cls_target = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        bbox_target = torch.zeros((4, self.grid_size, self.grid_size))
        centerness_target = torch.zeros((1, self.grid_size, self.grid_size))

        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            gi = int(cx / stride)
            gj = int(cy / stride)

            if gi < 0 or gj < 0 or gi >= self.grid_size or gj >= self.grid_size:
                continue

            l = cx - x
            t = cy - y
            r = (x + w) - cx
            b = (y + h) - cy

            bbox_target[:, gj, gi] = torch.tensor([l / stride, t / stride, r / stride, b / stride])
            cls_target[gj, gi] = label
            centerness_target[0, gj, gi] = (min(l, r) * min(t, b)) / (max(l, r) * max(t, b) + 1e-6)

        target = {
            "cls": cls_target,
            "bbox": bbox_target,
            "centerness": centerness_target,
            "image_id": image_id
        }

        return image, target


    
             