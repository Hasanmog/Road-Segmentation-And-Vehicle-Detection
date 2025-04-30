import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoadSegDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int, mode: str, augment=False, augmentations=None):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.augmentations = augmentations

        if mode in ["train", "val", "test"]:
            self.imgs = os.path.join(dataset_dir, f"{mode}/images")
            self.labels = os.path.join(dataset_dir, f"{mode}/labels")
        else:
            raise ValueError("Mode must be one of: 'train', 'val', 'test'")

        self.resize = T.Resize((img_size, img_size))

        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.ToTensor()

        self.images = sorted([f for f in os.listdir(self.imgs) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(self.labels) if f.endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.images) == len(self.masks), "Mismatch in images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.imgs, img_name)
        mask_path = os.path.join(self.labels, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.resize(image)
        mask = self.resize(mask)

        if self.augment and self.augmentations:
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.augmentations(image)
            torch.manual_seed(seed)
            mask = self.augmentations(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze(0)
        mask = (mask > 0.5).long()

        return {"pixel_values": image, "labels": mask}

