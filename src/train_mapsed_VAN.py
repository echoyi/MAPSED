from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model.mapsed import MAPSED
from src.model.vae.conv_vae import ConvVAE
from src.utils.calculate_seq_loss import calculate_seq_loss
from src.utils.load_data import load_seq_data, normalize_data


def save_progress(model, train_losses, train_nce, train_recon, train_losses_dicts_mean,
                  train_losses_dicts_var, test_losses_dicts_mean, test_losses_dicts_var):
    model_path = '../saved_models/VAN/mapsed-No-Contrast.torch'
    torch.save(model.state_dict(), model_path)

    metric_dict = {'training loss': train_losses,
                   'training nce': train_nce,
                   'training recon': train_recon}

    for me in metrics:
        metric_dict['train {}(mean)'.format(me)] = np.stack(train_losses_dicts_mean[me])
        metric_dict['train {}(std)'.format(me)] = np.stack(train_losses_dicts_var[me])
        metric_dict['test {}(mean)'.format(me)] = np.stack(test_losses_dicts_mean[me])
        metric_dict['test {}(std)'.format(me)] = np.stack(test_losses_dicts_var[me])

    df_result = pd.DataFrame(metric_dict)
    df_result.index.name = 'epochs'
    df_result.index = df_result.index + 1

    df_result.to_csv('../saved_models/VAN/mapsed-No-Contrast.csv')


device = torch.device('cuda')

seq_train, seq_test, _ = load_seq_data('VAN')
mean = seq_train.mean(axis=(0,1,3,4))
seq_train = normalize_data(seq_train, mean)
seq_test = normalize_data(seq_test, mean)


dataloader_train = DataLoader(seq_train, 32, True, drop_last=True)
dataloader_test = DataLoader(seq_test, 32, True, drop_last=True)
epochs = 51
train_losses = np.zeros(epochs)
train_nce = np.zeros(epochs)
train_recon = np.zeros(epochs)
use_same_input = False
# torch.manual_seed(2020)

if use_same_input:
    # only one data, check if the model can overfit
    # data = torch.stack([torch.tensor(seq_train[0]) for _ in range(128)])np.max(1, in_channels // 8)
    data = torch.tensor(seq_train[:32])
vae = ConvVAE(input_channels=4).to(device)
vae.load_state_dict(torch.load('../saved_models/VAN/VAE-VAN.torch'))
m = 5
n = 3

model = MAPSED(vae, latent_shape=(2, 5, 5),  m=m, n=n, lambda_contrast=0, contrast='inner').to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total # of trainable parameters: {}".format(pytorch_total_params))



metrics = ['MSE', 'MAE']
train_losses_dicts_mean = {me: [] for me in metrics}
train_losses_dicts_var = {me: [] for me in metrics}
test_losses_dicts_mean = {me: [] for me in metrics}
test_losses_dicts_var = {me: [] for me in metrics}

x_augs = []
for epoch in range(epochs):
    model.train()
    model.vae.eval()
    model.training = True
    running_train_loss = 0
    running_nce = 0
    running_recon = 0
    # running_diff = 0
    validate_loss = {me: [] for me in metrics}
    test_loss = {me: [] for me in metrics}
    for idx, seq in enumerate(dataloader_train):
        if use_same_input:
            seq = data
        # print("updating with data [{}/{}]".format(idx+1, len(dataloader_train)))
        with torch.no_grad():
            angle = np.random.choice([0, 90, 180, 270])
            #training transform
            train_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([angle, angle])])

            seq = seq.cuda().float()
            seq = train_tf(seq.view(-1,10,10)).view(seq.shape)

            indices = torch.randperm(m)
            x_aug = seq[:, :m].detach()[:, indices]
        optimizer.zero_grad()
        x = seq[:, :m]
        gt = seq[:, m:]
        y_pred, loss, nce_channel, recon_loss = model(x, gt_seq=gt, x_aug=x_aug)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.data
        running_nce += nce_channel
        running_recon += recon_loss
    train_losses[epoch] = running_train_loss / len(dataloader_train)
    train_nce[epoch] = running_nce / len(dataloader_train)
    train_recon[epoch] = running_recon / len(dataloader_train)

    with torch.no_grad():
        for me in metrics:
            validate_loss[me].append(calculate_seq_loss(y_pred, gt, me))

    model.eval()
    model.training = False
    with torch.no_grad():
        for _, seq in enumerate(dataloader_test):
            seq = seq.cuda().float()
            x = seq[:, :m]
            gt = seq[:, m:]
            y_pred, loss, _, _ = model(x, gt)
            for me in metrics:
                test_loss[me].append(calculate_seq_loss(y_pred, gt, me))
        for me in metrics:
            train_losses_dicts_mean[me].append(np.mean(validate_loss[me]))
            train_losses_dicts_var[me].append(np.std(validate_loss[me]))
            test_losses_dicts_mean[me].append(np.mean(test_loss[me]))
            test_losses_dicts_var[me].append(np.std(test_loss[me]))

    if epoch % 10 == 0:
        print("Time:{}, Running epoch [{}/{}], training loss:{}, channel nce:{}, recon:{}, train MSE loss:{},\
              test MSE loss:{}".format(
            datetime.now(),
            epoch + 1, epochs,
            train_losses[epoch],
            train_nce[epoch],
            train_recon[epoch],
            train_losses_dicts_mean['MSE'][epoch],
            test_losses_dicts_mean['MSE'][epoch]))
    if epoch % 30 == 0:
        save_progress(model, train_losses[:epoch + 1], train_nce[:epoch + 1],
                      train_recon[:epoch + 1], train_losses_dicts_mean, train_losses_dicts_var,
                      test_losses_dicts_mean, test_losses_dicts_var)
    scheduler.step(train_losses[epoch])
    # scheduler.step()
save_progress(model, train_losses, train_nce, train_recon, train_losses_dicts_mean,
              train_losses_dicts_var, test_losses_dicts_mean, test_losses_dicts_var)

print("DONE")
