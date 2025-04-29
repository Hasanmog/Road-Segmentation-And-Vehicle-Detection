import torch
from model.encoder.full_encoder import MultiScaleFusion
from model.seg_head.pred_head import Seg_Head
from model.det_head.detection import Det_Head
from model.model import SegDet


dummy_input = torch.randn(2, 3, 512, 512)


sam_ckpt_path = None 
swin_det_path = None  
swin_seg_path = None 


model = MultiScaleFusion(
    img_size=512,
    small_patch_size=8,
    large_patch_size=16,
    backbone="SWIN",
    backbone_freeze=True,
    sam_ckpt_path=sam_ckpt_path,
    swin_det_path=swin_det_path,
    swin_seg_path=swin_seg_path,
    dim=(256, 256),
    patches=(4, 4),
    depth=(2, 2),
    num_heads=(4, 4),
    mlp_ratio=(4, 4)
)


main_model = SegDet()

seg_head = Seg_Head()

det_head = Det_Head()

det , seg , local = model(dummy_input)
print("local" , local.shape)
masks , logits = seg_head(seg , local)

cls , bbox , center = det_head(det)


results = main_model(dummy_input)
print(results)


