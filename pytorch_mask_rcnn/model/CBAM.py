import torch.nn.functional as F
from torch import nn

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        # Channel attention module
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes//16, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes//16, in_planes, 1),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.sa = nn.Sequential(
            nn.Conv2d(in_planes, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x) * x
        sa = self.sa(ca)
        return sa