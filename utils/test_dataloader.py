from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from data.dataloader import RoadSegDataset  , VehicleDetDataset

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


dataset = RoadSegDataset(
    dataset_dir="/home/hasanmog/datasets/dataset_reduced",  
    img_size=512,
    mode="test"
)


loader = DataLoader(dataset, batch_size=1, shuffle=True)


for image, mask in loader:
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    image = denormalize(image.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    image_np = TF.to_pil_image(image[0].cpu())
    mask_np = TF.to_pil_image(mask[0].cpu())

   
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title("Input Image")
    ax[1].imshow(mask_np, cmap="gray")
    ax[1].set_title("Segmentation Mask")
    plt.tight_layout()
    plt.show()
    break  


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image


class_names = {0: "Car", 1: "Truck"}


dataset = VehicleDetDataset("/home/hasanmog/datasets/vedai", img_size=512, mode="train")
image, target = dataset[100]


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

image = denormalize(image.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image_vis = to_pil_image(image.cpu())

# Plot
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(image_vis)
print(f"bbox shape : {target['boxes'].shape}")
print(f"label shape: {target['labels'].shape}")
for box, label in zip(target['boxes'], target['labels']):
    x1, y1, x2, y2 = box.tolist()
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    class_name = class_names.get(label.item(), "Unknown")
    ax.text(x1, y1 - 5, class_name, color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5, pad=1))

ax.set_title(f"Image ID: {target['image_id']}")
plt.axis('off')
plt.tight_layout()
plt.show()
