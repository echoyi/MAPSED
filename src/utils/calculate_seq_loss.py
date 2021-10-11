import torch
import torch.nn.functional as F
import numpy as np


def calculate_loss(recon_hmaps, hmaps, loss_function):
    # (bs,l,w,h)
    loss = 0
    if loss_function == 'RMSE':
        loss = torch.sqrt(torch.mean(F.mse_loss(recon_hmaps, hmaps, reduction='none'), dim=(1, 2, 3)))
    if loss_function == 'MSE':
        loss = torch.mean(F.mse_loss(recon_hmaps, hmaps, reduction='none'), dim=(1, 2, 3))
    if loss_function == 'MAE':
        loss = torch.mean(F.l1_loss(recon_hmaps, hmaps, reduction='none'), dim=(1, 2, 3))
    if len(loss.shape) > 1:
        loss = torch.mean(loss, dim=(1, 2))
    return np.array(loss.data.cpu().detach())


def calculate_seq_loss(y_pred, y, loss_function):
    bs, l, c, w, h = y_pred.shape
    losses = []
    # category
    for c in range(c):
        loss = calculate_loss(y_pred[:, :, c], y[:, :, c], loss_function)
        losses.append(loss)
    losses = np.stack(losses, axis=1)
    return losses
