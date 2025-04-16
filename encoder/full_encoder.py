import torch.nn as nn
from encoder.img_encoder.img_encoder import Image_Encoders
from encoder.cross_attention.cross_attn import CrossAttnBlock



class MultiScaleFusion(nn.Module):
    def __init__(self , 
                 img_size: int = 512 ,
                 small_patch_size: int = 8 , 
                 large_patch_size: int = 16,
                 backbone_freeze: bool = True , 
                 enc_ckpt_path: str = None , 
                 dim: tuple = (256,256),  # 512 input img size
                 patches: tuple = None,
                 depth: tuple = (2 , 2),
                 num_heads: tuple = (4 , 4),
                 mlp_ratio: tuple = (4,4)):
        super().__init__()

        self.image_encoders = Image_Encoders(
                img_size=img_size,
                patch_small_size=small_patch_size , 
                patch_large_size=large_patch_size , 
                ckpt_path=enc_ckpt_path,
                freeze=backbone_freeze ,
)
        
        self.multi_scale = CrossAttnBlock(
            dim,
            patches,
            depth,
            num_heads,
            mlp_ratio
        )
        
        
    def forward(self , x):
        local_feat , global_feat = self.image_encoders(x) # [B , C1 , H1 , W1] , [B , C2 , H2 , W2]
        
        B1, C1, H1, W1 = local_feat.shape
        B2 , C2 , H2, W2 = global_feat.shape
        
        #flattening the outputs of the image encoders
        
        local_flatten = local_feat.flatten(2).transpose(1, 2) # [B , H1*W1 , C1]
        global_flatten = global_feat.flatten(2).transpose(1, 2) # [B , H2*W2 , C2]
        
        #cross features
        
        cross_feat = self.multi_scale([local_flatten , global_flatten])
        # cross_feat = [f[:, 1:] for f in cross_feat] # remove cls token
        
        return cross_feat
        
        
        

        
        