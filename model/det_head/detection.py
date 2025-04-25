import torch
import torch.nn as nn 



class Det_Head(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 4 + 1 + num_classes, 1)
        )
    
    def forward(self, x):
        return self.head(x)  # [B, 6, H, W] for 1 class   ; 6 --> 4 bbox coordinates , 1 object score , 2 class score



