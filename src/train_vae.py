import argparse
from datetime import datetime

import torch
from torchvision import transforms

from src.model.vae.conv_vae import ConvVAE
import numpy as np
from src.utils.load_data import load_data, normalize_data, recover_data
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from src.utils.calculate_loss import calculate_test_loss


def create_df(metrics, train_losses, reconstruct_losses, kl_losses,
              train_losses_dicts_mean, train_losses_dicts_var,
              test_losses_dicts_mean, test_losses_dicts_var):
    # save the metrics after all epochs
    metric_dict = {'training loss': train_losses,
                   'reconstruct loss': reconstruct_losses,
                   'kl divergence': kl_losses}

    for m in metrics:
        metric_dict['train {}(mean)'.format(m)] = np.stack(train_losses_dicts_mean[m])
        metric_dict['train {}(std)'.format(m)] = np.stack(train_losses_dicts_var[m])
        metric_dict['test {}(mean)'.format(m)] = np.stack(test_losses_dicts_mean[m])
        metric_dict['test {}(std)'.format(m)] = np.stack(test_losses_dicts_var[m])

    df_result = pd.DataFrame(metric_dict)
    df_result.index.name = 'epochs'
    df_result.index = df_result.index + 1
    return df_result


def train_vae(dataloader_train, dataloader_valid, dataset):
    # 4444
    torch.manual_seed(9999)
    vae = ConvVAE(4).to(device)
    pytorch_total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print("total # of trainable parameters: {}".format(pytorch_total_params))
    # vae.load_state_dict(torch.load('/home/amy/School'
    #                                '/Research/Convolutional-Transformer(Geographical)'
    #                                '/saved_models/VAE.torch'))
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)
    epochs = 191
    # VAN
    # epochs = 151

    train_losses = np.zeros(epochs)
    reconstruct_losses = np.zeros(epochs)
    kl_losses = np.zeros(epochs)
    bs = 32
    metrics = ['MSE', 'RMSE', 'MAE']

    train_losses_dicts_mean = {m: [] for m in metrics}
    train_losses_dicts_var = {m: [] for m in metrics}
    test_losses_dicts_mean = {m: [] for m in metrics}
    test_losses_dicts_var = {m: [] for m in metrics}

    time1 = datetime.now()
    for epoch in range(epochs):
        vae.train()
        vae.training = True
        running_train = 0
        running_reconstruct = 0
        running_kl = 0

        for idx, hmaps in enumerate(dataloader_train):
            with torch.no_grad():
                angle = np.random.choice([0, 90, 180, 270])
                train_tf = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation([angle, angle])
                     ])
                hmaps = hmaps.cuda().float()
                hmaps = train_tf(hmaps.view(-1, 10, 10)).view(hmaps.shape)
            # hmaps = hmaps.cuda().float()
            recon_hmaps, mu, logvar = vae(hmaps)
            loss, reconstruct, kl = vae.vae_loss(recon_hmaps, hmaps, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train += loss.data / bs
            running_reconstruct += reconstruct.data / bs
            running_kl += kl.data / bs
        train_losses[epoch] = running_train / len(dataloader_train)
        reconstruct_losses[epoch] = running_reconstruct / len(dataloader_train)
        kl_losses[epoch] = running_kl / len(dataloader_train)
        vae.eval()
        vae.training = False
        # validate model after training
        with torch.no_grad():
            for metric in metrics:
                train_mean, train_var = calculate_test_loss(vae, dataloader_train
                                                            , loss_function=metric)
                test_mean, test_var = calculate_test_loss(vae, dataloader_valid
                                                          , loss_function=metric)
                train_losses_dicts_mean[metric].append(train_mean)
                train_losses_dicts_var[metric].append(train_var)
                test_losses_dicts_mean[metric].append(test_mean)
                test_losses_dicts_var[metric].append(test_var)
                # if epoch%10 ==0:
                #     print("Epoch[{}/{}] Training {}: {:.3f}({:.3f}), Testing {} {:.3f}({:.3f}) ".format(epoch+1,
                #                               epochs, metric, train_mean,train_var, metric, test_mean, test_var))
        if epoch % 10 == 0:
            print('Timestamp: {}, Epoch {}/{}, loss: {}, KL: {}, reconstruct:{}'.format(
                datetime.now(), epoch + 1, epochs, train_losses[epoch], kl_losses[epoch],
                reconstruct_losses[epoch]))
        scheduler.step(train_losses[epoch])
    print("Time elapsed:{} ms".format(int((datetime.now() - time1).total_seconds() * 1000)))

    df_res = create_df(metrics, train_losses, reconstruct_losses, kl_losses,
                       train_losses_dicts_mean, train_losses_dicts_var,
                       test_losses_dicts_mean, test_losses_dicts_var)

    torch.save(vae.state_dict(), '../saved_models/{}/VAE-{}.torch'.format(dataset, dataset))
    df_res.to_csv('../saved_models/{}/VAE-{}.csv'.format(dataset, dataset))
    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="SF or VAN", type=str, default='SF')
    args = parser.parse_args()
    train, valid, _ = load_data(args.dataset)
    if args.dataset == 'SF':
        # max normalization
        normalizer = np.array([train[:, i].max() for i in range(4)])
    else:
        # mean normalization
        normalizer = train.mean(axis=(0, 2, 3))

    train = normalize_data(train, normalizer)
    valid = normalize_data(valid, normalizer)
    device = torch.device('cuda')

    dataloader_train = DataLoader(train, 32, shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(valid, 32, shuffle=True, drop_last=True)
    time1 = datetime.now()
    df_res = train_vae(dataloader_train, dataloader_valid, dataset=args.dataset)
    print(datetime.now() - time1)
    df_res['training loss'].plot()
    plt.show()
    df_res['kl divergence'].plot()
    plt.show()
    df_res['reconstruct loss'].plot()
    plt.show()
