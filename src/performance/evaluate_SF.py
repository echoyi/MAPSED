from datetime import datetime

import torch
from torch.utils.data import DataLoader
from src.model.mapsed import MAPSED
from src.model.vae.conv_vae import ConvVAE
from src.utils.calculate_seq_loss import calculate_seq_loss
from src.utils.load_data import load_seq_data, normalize_data_tensor, recover_data_tensor
import numpy as np
import pandas as pd


def evaluate(dataloader, model, metric, max):
    targets = []
    predicts = []
    for idx, seq in enumerate(dataloader):
        # print("updating with data [{}/{}]".format(idx+1, len(dataloader_train)))
        seq = seq.cuda().float()
        x = seq[:, :5]
        gt = seq[:, 5:]
        y_pred = model(normalize_data_tensor(x, max))[0].data
        predicts.append(recover_data_tensor(y_pred, max))
        targets.append(gt)
    predicts = torch.cat(predicts)
    targets = torch.cat(targets)
    loss = {m: None for m in metric}
    for m in metric:
        loss[m] = calculate_seq_loss(predicts, targets, m)
    loss_means = {m: loss[m].mean(axis=0) for m in metric}
    loss_vars = {m: loss[m].var(axis=0) for m in metric}
    loss_mean_and_var = {m: np.stack([loss_means[m], loss_vars[m]])
                         for m in metric}
    return loss, loss_mean_and_var


if __name__ == '__main__':
    device = torch.device('cuda')
    vae = ConvVAE(input_channels=4).to(device)
    vae.load_state_dict(torch.load('../../saved_models/SF/VAE-SF.torch'))
    m = 5
    n = 3

    model = MAPSED(vae, latent_shape=(2, 5, 5), m=m, n=n, lambda_contrast=0, contrast='L2').to(device)
    # model.load_state_dict(torch.load('../../saved_models/SF/MAPSED.torch'))
    # model.load_state_dict(torch.load('../../saved_models/SF/MAPSED-No-contrast.torch'))
    model.load_state_dict(torch.load('../../saved_models/SF/MAPSED-Inner-Product.torch'))
    model.training = False
    model.eval()
    vae.eval()

    seq_train, seq_valid, seq_test = load_seq_data('SF')
    max = np.array([seq_train[:, :, i].max() for i in range(4)])
    max = torch.tensor(max).cuda().float()
    mean = torch.tensor(seq_train.mean(axis=(0, 1, 3, 4))).cuda().float()
    normalizer = max
    print(len(seq_test))
    test_data_loader = DataLoader(seq_test, batch_size=32, shuffle=False, drop_last=True)
    train_data_loader = DataLoader(seq_train, batch_size=32, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(seq_valid, batch_size=32, shuffle=True, drop_last=True)
    metric = ['RMSE', 'MAE']

    loss_train, loss_train_mean_and_var = evaluate(train_data_loader, model, metric, normalizer)
    time1 = datetime.now()
    loss_test, loss_test_mean_and_var = evaluate(test_data_loader, model, metric, normalizer)
    print("Time elapsed:{} ms".format(int((datetime.now() - time1).total_seconds() * 1000)))
    loss_val, loss_val_mean_and_var = evaluate(valid_data_loader, model, metric, normalizer)


    index = pd.MultiIndex.from_product([['train', 'valid', 'test'],
                                        metric, ['mean', 'variance']],
                                       names=['metric', 'data source'
                                           , 'aggregation'])

    res = np.stack([np.stack(loss_train_mean_and_var.values()),
                    np.stack(loss_val_mean_and_var.values()),
                    np.stack(loss_test_mean_and_var.values())])
    print(res.shape)
    res = res.reshape(-1, 4)
    df = pd.DataFrame(data=res, index=index, columns=['type {}'.format(i + 1) for i in range(4)])
    df = df.round(4)
    print(df)

    # df.to_csv('SF/MAPSED.csv')
    # df.to_csv('SF/MAPSED-No-Contrast.csv')
    # df.to_csv('SF/MAPSED-Inner-Product.csv')