import torch.nn as nn

class BaseVAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(BaseVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.training = True

    def encode(self,x):
        raise NotImplementedError

    def decode(self,z):
        raise NotImplementedError

    def forward(x):
        pass