from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def load_data(root_path, category):
    path = '{}/{}-time-series.csv'.format(root_path, category)
    df = pd.read_csv(path, index_col=0)
    data = np.array([h.reshape(1, 10, 10) for h in df.values])
    return data


def load_data_channels(root_path, categories):
    data = []
    for c in categories:
        data.append(load_data(root_path, c))
    data = np.concatenate(data, axis=1)
    train, test = train_test_split(data)
    np.save('train', train)
    np.save('test', test)
    print(train.shape)


def create_dataloader(data, bs=32):
    return DataLoader(data, batch_size=bs, shuffle=True, drop_last=True)


def load_sequence_data(root_path, category, m=10, n=3, train_size=0.9, bs=32, step=7):
    path = '{}/{}-time-series-2016.csv'.format(root_path, category)
    df = pd.read_csv(path, index_col=0)
    data = np.array([h.reshape(1, 10, 10) for h in df.values])
    seq = []
    # m is observed, n is prediction
    for t0 in range(len(data) - (m + n) * step + 1):
        observed = data[t0:t0 + step * m:step]
        target = data[t0 + step * m:t0 + step * (m + n):step]
        seq.append(np.append(observed, target, axis=0))
    # seq shape: (N-(m+n)*step+1,m+n,1,10,10)
    # seq_train, seq_test = train_test_split(np.array(seq), train_size=train_size)
    return seq[:769]


#
def load_sequence_all(root_path, categories):
    data = []
    for c in categories:
        data.append(load_sequence_data(root_path, c))
    # L c w h
    data = np.concatenate(data, axis=2)
    print(data.shape)
    # seq_train, seq_test = train_test_split(data, train_size=0.8)
    return data


def load_seq_data(dataset='SF'):
    path = 'D:/MAPSED/MAPSED/data/{}'.format(dataset)
    seq_train = np.load('{}/seq_train.npy'.format(path))
    seq_valid = np.load('{}/seq_valid.npy'.format(path))
    seq_test = np.load('{}/seq_test.npy'.format(path))
    return seq_train, seq_valid, seq_test


def load_data(dataset='SF'):
    path = 'D:/MAPSED/MAPSED/data/{}'.format(dataset)
    train = np.load('{}/train.npy'.format(path))
    valid = np.load('{}/valid.npy'.format(path))
    test = np.load('{}/test.npy'.format(path))
    return train, valid, test


def normalize_data(hmaps, normalizer):
    shape = hmaps.shape
    output = hmaps.copy().astype(np.float64)
    if len(shape) == 5:
        N, L, c, w, h = shape
        output = output.reshape((-1, c, w, h))
    else:
        N, c, w, h = shape
    for i in range(c):
        output[:, i] = output[:, i] / normalizer[i]
    return normalizer.mean() * output.reshape(shape)


def recover_data(hmaps, normalizer):
    shape = hmaps.shape
    output = hmaps.copy().astype(np.float64)
    if len(shape) == 5:
        N, L, c, w, h = shape
        output = output.reshape((-1, c, w, h))
    else:
        N, c, w, h = shape
    for i in range(c):
        output[:, i] = output[:, i] * normalizer[i]
    return output.reshape(shape) / normalizer.mean()


def normalize_data_tensor(hmaps, normalizer):
    shape = hmaps.shape
    output = hmaps.clone()
    if len(shape) == 5:
        N, L, c, w, h = shape
        output = output.view((-1, c, w, h))
    else:
        N, c, w, h = shape
    for i in range(c):
        output[:, i] = output[:, i] / normalizer[i]
    return normalizer.mean() * output.view(shape)


def recover_data_tensor(hmaps, normalizer):
    shape = hmaps.shape
    output = hmaps.clone()
    if len(shape) == 5:
        N, L, c, w, h = shape
        output = output.reshape((-1, c, w, h))
    else:
        N, c, w, h = shape
    for i in range(c):
        output[:, i] = output[:, i] * normalizer[i]
    return output.reshape(shape) / normalizer.mean()