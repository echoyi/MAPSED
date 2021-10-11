from .base_vae import BaseVAE
import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvVAE(BaseVAE):
    def __init__(self, input_channels, latent_dim=(2,5,5)):
        super(ConvVAE, self).__init__(input_channels, latent_dim)
        self.encoder = nn.Sequential(
            # padding = same
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.Flatten()
        )

        self.conv1 = nn.Conv2d(64, self.latent_dim[0], kernel_size=1)
        self.conv2 = nn.Conv2d(64, self.latent_dim[0], kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim[0], 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            # force output to be positive
            nn.ReLU())

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.conv1(h), self.conv2(h)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        x_reconstruct = self.decoder(z)
        return x_reconstruct

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_reconstruct = self.decode(z)
        return x_reconstruct, mu, logvar

    def vae_loss(self, x_reconstruct, x, mu, logvar, beta=1, loss_fn='MSE'):
        if loss_fn == 'CE':
            reconstruct_loss = F.binary_cross_entropy(x_reconstruct, x, reduction='sum')
        else:
            # reconstruct_loss = F.mse_loss(x_reconstruct, x)
            reconstruct_loss = torch.sum(torch.sum(F.mse_loss(x_reconstruct, x, reduction='none'), dim=0))
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KL = torch.mean(-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KL = -0.5 * torch.sum(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=0))
        return reconstruct_loss + beta * KL, reconstruct_loss, KL
