import torch
import torch.nn as nn
from src.model.encoder_layer import EncLayer


class Encoder(nn.Module):

    def __init__(self, latent_shape=(8, 4, 4), m=5, contrast='L2'):
        super(Encoder, self).__init__()
        self.m = m
        self.contrast = contrast
        self.c, self.w, self.h = latent_shape

        # extraction layers
        self.layers = nn.ModuleList([EncLayer(input_shape=latent_shape, seq_len=m, contrast=contrast)
                                     for _ in range(2)])

    def forward(self, x, training, x_aug=None):
        s_aug = None
        if training:
            with torch.no_grad():
                s_aug = x_aug
        semantics = x
        dynamics = x
        merged = x
        nce = 0
        for i in range(2):
            semantics, dynamics, merged, nce_i = self.layers[i](semantics, dynamics, merged, s_aug=s_aug,
                                                                training=training)
            nce = nce_i
        return semantics, dynamics, merged, nce
