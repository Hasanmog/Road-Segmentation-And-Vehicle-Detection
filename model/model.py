import torch
import torch.nn as nn
from model.det_head.detection import Det_Head 
from model.seg_head.pred_head import Seg_Head
from model.encoder.full_encoder import MultiScaleFusion



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=256 , up_size = 64):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(up_size , up_size), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.upsample(x)





class SegDet(nn.Module):
    def __init__(self , 
                        img_size = 512 ,
                        small_patch_size = 8 ,
                        large_patch_size = 16 , 
                        backbone = "SWIN", 
                        backbone_freeze = True, 
                        sam_ckpt_path = None ,
                        swin_det_path = None , 
                        swin_seg_path = None,
                        local_feat_up = 64,
                        dim = 256):
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
                                                    mlp_ratio=(4, 4),
                                                )
        
        self.local_feat_up = local_feat_up
        self.seg_head = Seg_Head()
        
        self.det_head = Det_Head()
        self.upsample_local = Upsample(dim, dim, local_feat_up)
        self.upsample_det = Upsample(dim, dim, local_feat_up)
        self.upsample_global = Upsample(dim, dim, local_feat_up)

        
        
    def forward(self, x):
        det_feat, seg_feat, local_feat = self.encoder(x)

        
        local_feat_up = self.upsample_local(local_feat)
        det_feat = self.upsample_det(det_feat)
        seg_feat = self.upsample_global(seg_feat)
        mask_logits, masks = self.seg_head(seg_feat, local_feat_up)
        cls_logits, bbox, centerness = self.det_head(det_feat)

        return {
            "mask_logits": mask_logits,
            "masks": masks,
            "cls_logits": cls_logits,
            "bbox": bbox,
            "centerness": centerness
        }



        
        

