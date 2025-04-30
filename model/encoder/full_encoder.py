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

            if len(local_feat.shape) == 4:
                local_flatten = local_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
                global_flatten = global_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                local_flatten = local_feat
                global_flatten = global_feat
        else:
            if len(local_feat.shape) == 4:
                local_flatten = local_feat.flatten(2).transpose(1, 2)
                global_flatten = global_feat.flatten(2).transpose(1, 2)
            else:
                local_flatten = local_feat
                global_flatten = global_feat
            
            local_flatten = self.local_downsample(local_flatten)
            global_flatten = self.global_downsample(global_flatten)

        # Process through cross-attention block
        cross_feat = self.multi_scale([local_flatten, global_flatten])  # (local_cross, global_cross)

        # Split local and global cross attention
        local_cross, global_cross = cross_feat

        # Reshape everything back to [B, C, H, W]
        B0, N0, C0 = local_flatten.shape
        H0 = W0 = int(N0 ** 0.5)
        assert H0 * W0 == N0, "local_flatten N is not a perfect square."
        local_feat = local_flatten.reshape(B0, H0, W0, C0).permute(0, 3, 1, 2)  # [B, C, H, W]

        B1, N1, C1 = local_cross.shape
        H1 = W1 = int(N1 ** 0.5)
        assert H1 * W1 == N1, "local_cross N is not a perfect square."
        det_feat = local_cross.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2)  # [B, C, H, W]

        B2, N2, C2 = global_cross.shape
        H2 = W2 = int(N2 ** 0.5)
        assert H2 * W2 == N2, "global_cross N is not a perfect square."
        seg_feat = global_cross.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)  # [B, C, H, W]

        return det_feat, seg_feat, local_feat





    
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

    
        
        
        

        
        