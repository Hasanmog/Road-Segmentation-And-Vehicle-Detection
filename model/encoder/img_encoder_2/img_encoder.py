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
            num_classes=0
        )
        
        self.img_encoder_large = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_large_size,
            window_size=self.window_size(img_size, patch_large_size),
            num_classes=0
        )
        
        if ckpt_path_small is not None and ckpt_path_large is not None:
            self.load_pretrained_weights(ckpt_path_small, ckpt_path_large, freeze)
     
    def load_pretrained_weights(self, ckpt_path_small: str, ckpt_path_large: str, freeze: bool):
        ckpt_small = torch.load(ckpt_path_small, map_location='cuda')
        ckpt_large = torch.load(ckpt_path_large, map_location='cuda')
        self.img_encoder_small.load_state_dict(ckpt_small, strict=False)
        self.img_encoder_large.load_state_dict(ckpt_large, strict=False)

        if freeze:
            self.apply_freezing(self.img_encoder_small)
            self.apply_freezing(self.img_encoder_large)

    def apply_freezing(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = True

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

    
if __name__ == "__main__":

    # Create dummy input (batch of 2 images, 3 channels, 512x512)
    dummy_input = torch.randn(2, 3, 512, 512)
    ckpt_path_small = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/weights/mask_rcnn_swin_tiny_patch4_window7_1x.pth"
    ckpt_path_large = "/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/weights/upernet_swin_tiny_patch4_window7_512x512.pth"
    # Initialize the encoder
    encoder = Image_Encoders_Swin(
        img_size=512,
        patch_small_size=8,
        patch_large_size=16,
        ckpt_path_small=ckpt_path_small,
        ckpt_path_large=ckpt_path_large, 
        freeze=True
    )

    # Pass dummy input through the encoder
    local_features, global_features = encoder(dummy_input)

    # Print output shapes
    print(f"Local Features Shape: {local_features.shape}")
    print(f"Global Features Shape: {global_features.shape}")
