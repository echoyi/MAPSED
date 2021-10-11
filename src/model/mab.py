import torch
from torch.nn import Module, Conv2d, Softmax, Dropout
from src.model.bottleneck_layer import BottleneckLayer

#Ref from sagan
class SAM(Module):
    """ spatial attention module"""
    def __init__(self, input_shape):
        super(SAM, self).__init__()
        self.c, self.w, self.h = input_shape

        self.query_conv = Conv2d(in_channels=self.c, out_channels=self.c // 2, kernel_size=1, bias=False)
        self.key_conv = Conv2d(in_channels=self.c, out_channels=self.c // 2, kernel_size=1, bias=False)
        self.value_conv = Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=1, bias=False)
        self.dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        bs, c, h, w = x.shape
        proj_query = self.query_conv(x).view(bs, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(bs, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = self.dropout(attention)
        proj_value = self.value_conv(x).view(bs, -1, h * w)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, c, h, w)
        return out


class CAM(Module):
    """ channel attention module"""

    def __init__(self, input_shape):
        super(CAM, self).__init__()
        self.c, self.w, self.h = input_shape

        self.conv = Conv2d(self.c, self.c, 1, bias=False)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(0.1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        bs, c, h, w = x.shape
        x_conv = self.conv(x).view(bs, c, -1)
        proj_query = x_conv
        proj_key = x_conv.permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.dropout(self.softmax(energy))
        proj_value = x.view(bs, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(bs, c, h, w)
        return out


# MAB block
class MAB(Module):
    def __init__(self, input_shape=None):
        super(MAB, self).__init__()
        self.channel = CAM(input_shape)
        self.spatial = SAM(input_shape)
        self.conv = BottleneckLayer(input_shape[0] * 2, input_shape[0])

    def forward(self, x):
        bs, c, w, h = x.shape
        output = torch.cat([self.channel(x), self.spatial(x)], dim=-3)
        output = self.conv(output.view(-1, 2 * c, w, h))
        output = x + output

        return output
