import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np


class BetaVAEConv(BaseVAE):

    num_iter = 0
    def __init__(self, 
                 latent_dim=10,
                 beta = 1,
                 img_size = (1,64, 64),
                 latent_dist = 'bernoulli'):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(BetaVAEConv, self).__init__()

        # Layer parameters
        hid_channels = 32
        hid_channels2 = 64
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.beta = beta
        self.img_size = img_size
        self.latent_dist = latent_dist
        # Shape required to start transpose convs
        
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels2, kernel_size, **cnn_kwargs)

        self.conv_64 = nn.Conv2d(hid_channels2, hid_channels2, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.line1 = nn.Linear(hid_channels2*kernel_size*kernel_size, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)



        self.lind1 = nn.Linear(latent_dim, hidden_dim)
        self.lind2 = nn.Linear(hidden_dim, hidden_dim)
        self.lind3 = nn.Linear(hidden_dim, hid_channels*kernel_size*kernel_size)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

    def encode(self, input):
        batch_size = input.size(0)
        
        x = torch.relu(self.conv1(input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.line1(x))

        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

    def decode(self, input):

        batch_size = input.size(0)
        
        x = torch.relu(self.lind1(input))
        x = torch.relu(self.lind2(x))
        x = torch.relu(self.lind3(x))
        x = x.view(batch_size, 32, 4, 4)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon, x, mu, log_var):
        self.num_iter += 1
        batch_size = x.size(0)

        if self.latent_dist == 'bernoulli':
            recon_loss =F.binary_cross_entropy(recon, x, reduction='sum')
        elif self.latent_dist  == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
            recon_loss = F.mse_loss(recon * 255, x * 255, reduction="sum") / 255

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

        
        loss = recon_loss + self.beta  * kld_loss
        return loss


    #smaple form latent sapce
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    #generate reconstructed image
    def generate(self, x):
        return self.forward(x)[0]