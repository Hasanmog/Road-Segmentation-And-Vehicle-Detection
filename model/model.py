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
        det_feat , seg_feat , local_feat = self.encoder(x)

        mask_logits , masks = self.seg_head(seg_feat , local_feat)
        cls_logits , bbox , centerness = self.det_head(det_feat)

        results = {
            "mask_logits" : mask_logits ,
            "masks" : masks , 
            "cls_logits" : cls_logits , 
            "bbox" : bbox , 
            "centerness" : centerness     
        }

        return results


        
        

