import torch.nn as nn
from model.encoder.img_encoder.img_encoder import Image_Encoders_SAM
from model.encoder.cross_attention.cross_attn import CrossAttnBlock
from model.encoder.img_encoder_2.img_encoder import Image_Encoders_Swin


class MultiScaleFusion(nn.Module):
    def __init__(self , 
                 img_size: int = 512 ,
                 small_patch_size: int = 8 , 
                 large_patch_size: int = 16,
                 backbone: str = "SWIN" ,
                 backbone_freeze: bool = True , 
                 sam_ckpt_path: str = None ,
                 swin_det_path: str = None ,
                 swin_seg_path: str = None , 
                 dim: tuple = (256,256),  # 512 input img size
                 patches: tuple = None,
                 depth: tuple = (2 , 2),
                 num_heads: tuple = (4 , 4),
                 mlp_ratio: tuple = (4,4)):
        super().__init__()
        self.backbone = backbone
        if backbone == "SAM":
            self.image_encoders = Image_Encoders_SAM(
                    img_size=img_size,
                    patch_small_size=small_patch_size , 
                    patch_large_size=large_patch_size , 
                    ckpt_path=sam_ckpt_path,
                    freeze=backbone_freeze ,
             )
            
        elif  backbone == "SWIN":
          self.image_encoders = Image_Encoders_Swin(
                    img_size=512,
                    patch_small_size=8,
                    patch_large_size=16,
                    ckpt_path_small=swin_det_path,
                    ckpt_path_large=swin_seg_path, 
                    freeze=True
             ) 
        
        self.multi_scale = CrossAttnBlock(
            dim,
            patches,
            depth,
            num_heads,
            mlp_ratio
        )
        
        self.local_downsample = nn.Linear(768, 256)
        self.global_downsample = nn.Linear(768, 256)  
        
    def forward(self, x):
        local_feat, global_feat = self.image_encoders(x)

        if self.backbone == "SAM":
            B1, C1, H1, W1 = local_feat.shape
            B2, C2, H2, W2 = global_feat.shape

            # Flatten only if the tensor is 4D (i.e., [B, C, H, W])
            if len(local_feat.shape) == 4:
                local_flatten = local_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
                global_flatten = global_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                local_flatten = local_feat
                global_flatten = global_feat
        else:
            # apply flattening or skip based on tensor shape
            if len(local_feat.shape) == 4:
                local_flatten = local_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
                global_flatten = global_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                local_flatten = local_feat
                global_flatten = global_feat
            
            # Downsample from 768 to 256 if necessary
            local_flatten = self.local_downsample(local_flatten)  # Apply downsampling (768 -> 256)
            global_flatten = self.global_downsample(global_flatten)  # Apply downsampling (768 -> 256)
        cross_feat = self.multi_scale([local_flatten, global_flatten])  # Process through the multi-scale block

        return cross_feat




    
if __name__ == "__main__":
    import torch

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

    
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")

    
        
        
        

        
        