from model import SegDet
def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

model = SegDet()
print("Encoder:", count_params(model.encoder))
print("Seg Head:", count_params(model.seg_head))
print("Det Head:", count_params(model.det_head))


model = SegDet(ckpt_path="/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/weights/sam_vit_b_01ec64.pth")

print("Trainable Parameters Only:")
print("Encoder:", count_trainable_params(model.encoder))
print("Seg Head:", count_trainable_params(model.seg_head))
print("Det Head:", count_trainable_params(model.det_head))

total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_trainable:,}")

