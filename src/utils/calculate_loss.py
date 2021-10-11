import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(self, ctx, x, scale):
        self.save_for_backward = scale
        return x

    @staticmethod
    def backward(self, grad_outputs):
        scale = self.save_for_backward
        return scale * grad_outputs


class GradScaler(nn.Module):
    def __init__(self, scale=0.1):
        super(GradScaler, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x

    def backward(self, x):
        output = GradScale(x, self.scale)


def calculate_loss(pred, gt, loss_function, reduction='mean'):
    loss = 0
    if loss_function == 'RMSE':
        loss = torch.sqrt(F.mse_loss(pred, gt, reduction=reduction))
    if loss_function == 'MSE':
        loss = F.mse_loss(pred, gt, reduction=reduction)
    if loss_function == 'MAE':
        loss = F.l1_loss(pred, gt, reduction=reduction)
    if loss_function == 'CE':
        loss = F.binary_cross_entropy(pred, gt, reduction=reduction)
    return loss


def calculate_test_loss(vae, data_loader, loss_function='RMSE', category=None):
    loss = np.array([])
    for idx, hmaps in enumerate(data_loader):
        if category is not None:
            hmaps = hmaps[:, category].unsqueeze(1).cuda().float()
        else:
            hmaps = hmaps.cuda().float()
        recon_hmaps, _, _ = vae(hmaps)
        loss = np.append(loss, calculate_loss(hmaps, recon_hmaps, loss_function).data.cpu())
    return round(np.mean(loss), 4), round(np.std(loss), 4)


def get_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    return mask.cuda()


def calculate_content_nce(z, z_aug, function='L2'):
    loss = 0
    bs = z.shape[0]
    # 2*bs, *
    latent_seq = torch.cat([z, z_aug], dim=0).view(2 * bs, -1)
    # shape:bs
    positive_scores = torch.mean(F.mse_loss(z.view(bs, -1), z_aug.view(bs, -1), reduction='none'), dim=-1)
    if function == 'L1L2':
        positive_scores += torch.mean(F.l1_loss(z.view(bs, -1), z_aug.view(bs, -1), reduction='none'), dim=-1)
    # force latent feature maps of the same seq to be alike
    # positive samples: in ths same seq; negative samples: different bs
    # F.relu(self_scores - most_alike_others + constant)
    # calculate self scores:
    t1 = torch.stack([latent_seq for _ in range(2 * bs)], dim=0)
    t2 = torch.stack([latent_seq for _ in range(2 * bs)], dim=1)
    scores = torch.mean(F.mse_loss(t1, t2, reduction='none'), dim=-1)
    if function == 'L1L2':
        scores += torch.mean(F.l1_loss(t1, t2, reduction='none'), dim=-1)
    assert scores.shape == (2 * bs, 2 * bs), "shape {} is not as expected".format(scores.shape)
    assert scores[0, 0] == 0, "self score should be zero"
    assert scores[0, bs] == positive_scores[0], "positive pair scores"
    mask = get_mask(batch_size=bs).detach() * scores.max().detach()
    most_alike_others, _ = torch.min(scores + mask, 0)
    most_alike_others = most_alike_others[:bs]
    most_alike_others = GradScaler()(most_alike_others)
    loss = loss + torch.mean(F.relu(positive_scores - most_alike_others + 2))
    return loss
