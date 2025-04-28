import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Det_Head(nn.Module):
    def __init__(self, in_channels=256, num_classes=2, num_convs=4, prior_prob=0.01):
        super().__init__()

        # Shared stem: Deeper and normalized
        stem = []
        for _ in range(num_convs):
            stem.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(in_channels))
            stem.append(nn.ReLU(inplace=True))
        self.shared_conv = nn.Sequential(*stem)

        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

        # Regression branch
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

        # Centerness branch
        self.centerness_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        # Initialize classification bias (helps faster learning)
        self._init_bias(prior_prob)

    def _init_bias(self, prior_prob):
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_branch[-1].bias, bias_value)

    def forward(self, x):
        features = self.shared_conv(x)

        cls_logits = self.cls_branch(features)
        bbox_reg = self.reg_branch(features)
        centerness = self.centerness_branch(features)

        return cls_logits, bbox_reg, centerness







