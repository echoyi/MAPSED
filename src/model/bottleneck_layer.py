import torch.nn as nn
import torch
import torch.nn.functional as F


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        self.conv3 = nn.Conv2d(in_channels * 2, out_channels, 1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.in_channels == self.out_channels:
            out = x + out
        return out

class BottleneckLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, 1,bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels * 2)
        self.conv2 = nn.Conv3d(in_channels * 2, in_channels * 2, 3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels * 2)
        self.conv3 = nn.Conv3d(in_channels * 2, out_channels, 1,bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.in_channels == self.out_channels:
            out = x + out
        return out


