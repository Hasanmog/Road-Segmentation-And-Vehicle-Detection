import torch
import torch.nn as nn
from model.det_head.detection import Det_Head 
from model.seg_head.pred_head import Seg_Head
from model.encoder.full_encoder import MultiScaleFusion


class SegDet(nn.Module):
    def __init__(self , 
                        img_size = 512 ,
                        small_patch_size = 8 ,
                        large_patch_size = 16 , 
                        backbone = "SWIN", 
                        backbone_freeze = True, 
                        sam_ckpt_path = None ,
                        swin_det_path = None , 
                        swin_seg_path = None):
        super().__init__()
        
        self.encoder = MultiScaleFusion(
                                                    img_size=img_size,
                                                    small_patch_size=small_patch_size,
                                                    large_patch_size=large_patch_size,
                                                    backbone=backbone,
                                                    sam_ckpt_path=sam_ckpt_path,
                                                    swin_det_path=swin_det_path,
                                                    swin_seg_path=swin_seg_path,
                                                    backbone_freeze=backbone_freeze,
                                                    dim=(256, 256),
                                                    depth=(4, 4),
                                                    num_heads=(4, 4),
                                                    mlp_ratio=(4, 4)
                                                )
        
        self.seg_head = Seg_Head()
        
        self.det_head = Det_Head()
        
        
        
    def forward(self, x):
        local_cross, global_cross = self.encoder(x)

        B1, N1, C1 = local_cross.shape
        H1 = W1 = int(N1 ** 0.5)
        det_feat = local_cross.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2)

        B2, N2, C2 = global_cross.shape
        H2 = W2 = int(N2 ** 0.5)
        seg_feat = global_cross.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)

        masks = self.seg_head(seg_feat)
        detections = self.det_head(det_feat)

        bbox = detections[:, 0:4, :, :]                # [B, 4, H, W]
        obj_score = detections[:, 4:5, :, :]           # [B, 1, H, W]
        class_score = detections[:, 5:, :, :]          # [B, 2, H,_


        results = {
            "masks": masks,
            "bbox": bbox,
            "obj_score": obj_score,
            "class_score": class_score
        }

        return results


        
        

