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
        self.ca_weights = self.ca(x)
        ca = self.ca_weights * x
        self.sa_weights = self.sa(ca)
        return ca * self.sa_weights

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # n,c,h,w
        _, _, h, w, = x.size()

        # n,c,h,w -> c,h,1 and c,1,w
        x_h = torch.mean(x, dim=3, keepdim = True).permute(0,1,3,2) 
        x_w = torch.mean(x, dim=2, keepdim = True)

        #n,c,1,w  cat n,c,1,h  -> n, c, 1, w+c
        #n,c,1,w  cat n,c,1,h  -> n, c/r, 1, w+c 简化层数
        x_cat_conv_relu = self.relu(self.bn(self.conv_1(torch.cat((x_h,x_w),3))))

        # split : n, c/r, 1, h and n, c/r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h,w],3)

        #n, c/r, h, 1 and n, c/r, 1, w
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0,1,3,2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x
