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

    # def _resize_bbox_to_xyxy(self, bbox, orig_w, orig_h):
    #     scale_x = self.img_size / orig_w
    #     scale_y = self.img_size / orig_h
    #     x, y, w, h = bbox
    #     x1 = x * scale_x
    #     y1 = y * scale_y
    #     x2 = (x + w) * scale_x
    #     y2 = (y + h) * scale_y
    #     return [x1, y1, x2, y2]

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

        obj_target = torch.zeros((self.grid_size, self.grid_size))
        cls_target = torch.zeros((self.grid_size, self.grid_size))
        bbox_target = torch.zeros((4, self.grid_size, self.grid_size))  # cx, cy, w, h

       
        id_map = {1: 0, 2: 1}  
        stride = self.img_size / self.grid_size

        for ann in anns:
            if ann["category_id"] not in id_map:
                continue

            label = id_map[ann["category_id"]]

            
            x, y, w, h = ann["bbox"]

            x *= self.img_size / original_w
            y *= self.img_size / original_h
            w *= self.img_size / original_w
            h *= self.img_size / original_h

            cx = x + w / 2
            cy = y + h / 2

            gi = int(cx / stride)
            gj = int(cy / stride)

            if gi < 0 or gj < 0 or gi >= self.grid_size or gj >= self.grid_size:
                continue


            cx_norm = (cx / stride) - gi  
            cy_norm = (cy / stride) - gj
            w_norm = w / stride
            h_norm = h / stride

            obj_target[gj, gi] = 1
            cls_target[gj, gi] = label
            bbox_target[:, gj, gi] = torch.tensor([cx_norm, cy_norm, w_norm, h_norm])

        target = {
            "obj": obj_target,
            "labels": cls_target,
            "boxes": bbox_target,
            "image_id": image_id
        }

        return image, target

    
    
               
if __name__ == "__main__":
    
    import torch
    from torchvision.transforms.functional import to_pil_image
    from PIL import Image, ImageDraw
    import os
    import matplotlib.pyplot as plt

    # De-normalize image tensor
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor



    # --- SEGMENTATION VISUALIZATION ---

    seg_dataset = RoadSegDataset(
        dataset_dir="/home/hasanmog/datasets/dataset_reduced",
        mode="train",
        img_size=512
    )
    
        # For both segmentation and detection images:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image, mask = seg_dataset[0]  # image: [3, H, W], mask: [1, H, W]
    image = denormalize(image.clone(), mean, std).clamp(0, 1)
    image_pil = to_pil_image(image)  # Convert to PIL
    mask_np = (mask.squeeze(0).numpy() * 255).astype('uint8')
    mask_img = Image.fromarray(mask_np).convert("L")

    # Create overlay
    image_with_mask = image_pil.copy().convert("RGBA")
    mask_img = mask_img.convert("RGBA")
    mask_img.putalpha(128)
    seg_overlay = Image.alpha_composite(image_with_mask, mask_img).convert("RGB")

    # --- DETECTION VISUALIZATION ---

    det_dataset = VehicleDetDataset(
        dataset_dir="/home/hasanmog/datasets/vedai",
        mode="train",
        img_size=512
    )
    image_det, det_target = det_dataset[0]
    image_det = denormalize(image_det.clone(), mean, std).clamp(0, 1)
    obj, labels, boxes = det_target["obj"], det_target["labels"], det_target["boxes"]
    image_det_pil = to_pil_image(image_det).convert("RGB")

    # Find object to draw
    nonzero = obj.nonzero(as_tuple=True)
    if len(nonzero[0]) > 0:
        i, j = nonzero[0][0].item(), nonzero[1][0].item()
        cx, cy, w, h = boxes[:, i, j]
        stride = 512 / 64

        # Convert to pixel coords
        cx_abs = (j + cx.item()) * stride
        cy_abs = (i + cy.item()) * stride
        w_abs = w.item() * stride
        h_abs = h.item() * stride
        x1 = cx_abs - w_abs / 2
        y1 = cy_abs - h_abs / 2
        x2 = cx_abs + w_abs / 2
        y2 = cy_abs + h_abs / 2

        draw = ImageDraw.Draw(image_det_pil)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Label: {int(labels[i, j].item())}", fill="red")

    # --- PLOT BOTH RESULTS ---

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(seg_overlay)
    plt.title("Segmentation Overlay")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_det_pil)
    plt.title("Detection Visualization")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



    
    
             