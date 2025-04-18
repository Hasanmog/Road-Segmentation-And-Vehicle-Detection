import torch
import torch.nn as nn


class Seg_Head(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.head_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.head_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.head_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=1),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        )
        
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.head_layer1(x)
        x = self.head_layer2(x)
        x = self.head_layer3(x)
        x = self.activation(x)
        return x
