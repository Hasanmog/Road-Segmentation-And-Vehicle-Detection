import torch
import torch.nn as nn 


class Det_Head(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 4 + 1 + num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.head(x)




