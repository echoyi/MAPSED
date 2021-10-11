import torch
import torch.nn as nn

from src.model.mab import MAB
from src.model.bottleneck_layer import BottleneckLayer

from src.utils.calculate_loss import calculate_content_nce
from src.utils.contrastive_loss import NTXentLoss


class EncLayer(nn.Module):

    def __init__(self, input_shape, seq_len, contrast='L2'):
        super(EncLayer, self).__init__()
        self.c, self.w, self.h = input_shape
        self.m = seq_len
        self.contrast = contrast
        # semantics extraction
        self.s_extraction = MAB((self.c, self.w * self.m, self.h))
        # dynamics extraction
        self.d_extraction = MAB((self.c * self.m, self.w, self.h))

        self.merge = BottleneckLayer(2 * self.c, self.c)
        self.conv = BottleneckLayer(self.c, self.c)
        if contrast == 'inner':
            self.nceLoss = NTXentLoss(torch.device('cuda'), temperature=1, use_cosine_similarity=False)

    def forward(self, semantics, dynamics, merged, s_aug=None, training=True):
        nce = 0
        semantics = torch.cat([semantics[:, i] for i in range(self.m)], dim=-2)
        dynamics = torch.cat([dynamics[:, i] for i in range(self.m)], dim=1)
        semantics = self.s_extraction(semantics)
        if training:
            with torch.no_grad():
                c_aug = torch.cat([s_aug[:, i] for i in range(self.m)], dim=-2)
                c_aug = self.s_extraction(c_aug)
            nce = self._nce(semantics, c_aug)
        dynamics = self.d_extraction(dynamics)
        # transform back into original shapes
        semantics = torch.stack([semantics[:, :, i * self.w:(i + 1) * self.w]
                                 for i in range(self.m)], dim=1)
        dynamics = torch.stack([dynamics[:, self.c * i: self.c * (i + 1)]
                                for i in range(self.m)], dim=1)
        merged = self._merge(semantics, dynamics, merged)
        return semantics, dynamics, merged, nce

    def _nce(self, semantics, s_aug):
        if self.contrast == 'cosine':
            nce = self.nceLoss(semantics, s_aug)
        else:
            nce = calculate_content_nce(semantics, s_aug, self.contrast)
        return nce

    def _merge(self, semantics, dynamics, merged):
        merged = self.conv(merged.contiguous().view(-1, self.c, self.w, self.h))
        merged = merged.contiguous().view((-1, self.m, self.c, self.w, self.h))
        value = torch.cat([semantics, dynamics], dim=2).contiguous().view((-1, self.c * 2, self.w, self.h))
        value = self.merge(value)
        value = value.contiguous().view(merged.shape) + merged
        return value
