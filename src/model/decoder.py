import torch
import torch.nn as nn
from src.model.bottleneck_layer import BottleneckLayer, BottleneckLayer3D
from src.model.mab import MAB


class Decoder(nn.Module):
    def __init__(self, latent_shape=(4, 10, 10), m=10, n=3):
        super(Decoder, self).__init__()
        self.c, self.w, self.h = latent_shape
        self.m = m
        self.n = n
        self.fusion = nn.Sequential(BottleneckLayer3D(m, n),
                                    BottleneckLayer3D(n, n))
        self.d_attention = MAB((self.c * self.n, self.w, self.h))
        self.s_attention = MAB((self.c, self.w * self.n, self.h))

        self.decode = BottleneckLayer(self.c, self.c)

    def forward(self, z):
        bs, seq_len, c, w, h = z.shape
        output = self.fusion(z)
        output = output.view(bs, self.n, c, w, h)
        output = self.d_attention(self._reshape_for_dynamics(output, forward=True))
        output = self._reshape_for_dynamics(output, forward=False)
        output = self.s_attention(self._reshape_for_semantics(output, forward=True))
        output = self._reshape_for_semantics(output, forward=False)
        output = output.contiguous().view((bs, self.n, c, w, h))
        return output

    def _reshape_for_semantics(self, x, forward=True):
        if forward:
            x = torch.cat([x[:, i] for i in range(self.n)], dim=-2)
        else:
            x = torch.stack([x[:, :, i * self.w:(i + 1) * self.w] for i in range(self.n)], dim=1)
        return x

    def _reshape_for_dynamics(self, x, forward=True):
        if forward:
            x = torch.cat([x[:, i] for i in range(self.n)], dim=1)
        else:
            x = torch.stack([x[:, self.c * i: self.c * (i + 1)] for i in range(self.n)], dim=1)
        return x
