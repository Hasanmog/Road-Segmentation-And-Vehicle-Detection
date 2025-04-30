import torch
import torch.nn as nn
from typing import Optional
from model.encoder.img_encoder_2.swin_transformer_v2 import SwinTransformerV2 


class Image_Encoders_Swin(nn.Module):
    def __init__(self, img_size: int = 512, 
                 in_channels: int = 3, 
                 patch_small_size: int = 8, 
                 patch_large_size: int = 16, 
                 ckpt_path_small: Optional[str] = None, 
                 ckpt_path_large: Optional[str] = None, 
                 freeze: bool = True):
        super().__init__()
        
        self.patch_small_size = patch_small_size 
        self.patch_large_size = patch_large_size 
        self.in_channels = in_channels
        self.img_size = img_size
        
        self.img_encoder_small = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_small_size,
            window_size=self.window_size(img_size, patch_small_size),
            num_classes=0,
            ape = True ,  #positional embedding
        )
        
        self.img_encoder_large = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_large_size,
            window_size=self.window_size(img_size, patch_large_size),
            num_classes=0,
            ape = True #positional embedding
        )
        
        if ckpt_path_small is not None and ckpt_path_large is not None:
            self.load_pretrained_weights(ckpt_path_small, ckpt_path_large, freeze)
     
    def load_pretrained_weights(self, ckpt_path_small: str, ckpt_path_large: str, freeze: bool):
        ckpt_small = torch.load(ckpt_path_small, map_location='cpu')
        ckpt_large = torch.load(ckpt_path_large, map_location='cpu')
        self.img_encoder_small.load_state_dict(ckpt_small, strict=False)
        self.img_encoder_large.load_state_dict(ckpt_large, strict=False)

        if freeze:
            self.apply_freezing(self.img_encoder_small)
            self.apply_freezing(self.img_encoder_large)

    def apply_freezing(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False

        for param in encoder.patch_embed.parameters():
            param.requires_grad = True

        for idx, layer in enumerate(encoder.layers):
            if idx % 2 == 0:
                for param in layer.parameters():
                    param.requires_grad = True

    @staticmethod
    def window_size(img_size, patch_size, max_good_size=16):
        patches_per_side = img_size // patch_size
        best_window_size = 1
        for w in range(1, patches_per_side + 1):
            if patches_per_side % w == 0 and w <= max_good_size:
                best_window_size = w
        return best_window_size

    def forward(self, x):
        local_features = self.img_encoder_small(x)
        global_features = self.img_encoder_large(x)
        return local_features, global_features

if __name__ == '__main__':

    dummy = torch.randn((1 , 3 , 512 , 512))

    model = Image_Encoders_Swin()
    
    output = model(dummy)

    print(output[0].shape , output[1].shape) # torch.Size([1, 64, 768]) torch.Size([1, 16, 768])