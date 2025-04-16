import torch
import torch.nn as nn
from typing import Optional
from segment_anything.modeling.image_encoder import ImageEncoderViT


class Image_Encoders(nn.Module):
    def __init__(self , img_size: int = 512 , 
                 in_channels: int = 3 , 
                 patch_small_size: int= 8 , 
                 patch_large_size: int= 16, 
                 ckpt_path: Optional[str] = None , 
                 freeze: bool = True):
        super().__init__()
        
        self.patch_small_size = patch_small_size 
        self.patch_large_size = patch_large_size 
        self.in_channels = in_channels
        self.img_size = img_size
        
        self.img_encoder_small = ImageEncoderViT(img_size = img_size ,
                                                 in_chans = in_channels , 
                                             patch_size = patch_small_size , 
                                             )
        
        self.img_encoder_large = ImageEncoderViT(img_size = img_size , 
                                                 in_chans = in_channels , 
                                                 patch_size = patch_large_size)
        
        if ckpt_path is not None:
            self.load_pretrained_weights(ckpt_path, freeze)
     
    def load_pretrained_weights(self, ckpt_path: str , freeze):
        """Load pretrained SAM encoder weights from a checkpoint file."""
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
    
        encoder_state = state_dict.get("image_encoder", state_dict)

    
        self.img_encoder_small.load_state_dict(encoder_state, strict=False)
        self.img_encoder_large.load_state_dict(encoder_state, strict=False)
        
        if freeze:
            for p in self.img_encoder_small.parameters():
                p.requires_grad = False
            for p in self.img_encoder_large.parameters():
                p.requires_grad = False

             
    def forward(self , x):
        
        local_features = self.img_encoder_small(x)
        global_features = self.img_encoder_large(x)
    
        return local_features , global_features