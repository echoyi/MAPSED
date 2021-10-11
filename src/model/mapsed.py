import torch.nn as nn
import torch
import torch.nn.functional as F
from src.model.encoder import Encoder
from src.model.decoder import Decoder


class MAPSED(nn.Module):

    def __init__(self, vae, latent_shape=(4,3,3), m=5, n=3, lambda_contrast=1, contrast ='L2',lambda_MAE=1):
        super(MAPSED, self).__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_MAE = lambda_MAE
        self.vae = vae
        self.pred_seq = n
        vae.training = False
        self.debug = False
        self.training = True
        self.seq_len = m
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        self.encoder = Encoder(latent_shape, m, contrast=contrast)
        self.decoder = Decoder(latent_shape, m=m, n=n)

    def forward(self, x, gt_seq=None, x_aug=None):
        loss = 0
        recon_loss = 0
        feature_maps_seq = self._encode_feature_seq(x)
        feature_maps_seq_aug = None
        if self.training:
            feature_maps_seq_aug = self._encode_feature_seq(x_aug)

        semantics, dynamics, z, nce = self.encoder(feature_maps_seq, self.training,
                                                   x_aug=feature_maps_seq_aug)
        decoded_maps = self.decoder(z)
        pred_seq = []
        for i in range(self.pred_seq):
            pred_seq.append(self.vae.decode(decoded_maps[:, i]))
        pred_seq = torch.stack(pred_seq, dim=1)
        if self.training:
            recon_loss = self.loss_fn(pred_seq, gt_seq, metric='L1L2')
            loss = self.lambda_contrast * nce + recon_loss
        if self.debug:
            return pred_seq, loss, nce, recon_loss, decoded_maps
        else:
            return pred_seq, loss, nce, recon_loss

    def loss_fn(self, pred, target, metric='MSE', per_frame=False):
        loss = 0
        # sum over shape and mean over bs
        if metric == 'MSE':
            loss = loss + torch.mean(
                torch.sum(F.mse_loss(pred, target, reduction='none'), dim=1))
        else:
            loss = loss + torch.mean(
                torch.sum(self.lambda_MAE*F.l1_loss(pred, target, reduction='none') + F.mse_loss(pred, target, reduction='none'),
                          dim=1))
        if per_frame:
            loss = loss / pred.shape[1]
        return loss

    def _encode_feature_seq(self, x):
        feature_maps_seq = []
        for i in range(self.seq_len):
            z, mu, var = self.vae.encode(x[:, i])
            feature_maps_seq.append(z)
        # (m, bs, c, w, h) ==> (bs, m, c, w, h)
        feature_maps_seq = torch.transpose(torch.stack(feature_maps_seq, dim=0), 0, 1)
        return feature_maps_seq
