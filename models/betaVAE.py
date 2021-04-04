import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BetaVAE(BaseVAE):

    num_iter = 0
    def __init__(self,
                latent_dim,
                beta = 1,
                loss_type = 'H'):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_type = loss_type

        input_dim = 4096

        self.line1 = nn.Linear(input_dim, 1200)
        self.line2 = nn.Linear(1200, 1200)
        self.mu_logvar_gen = nn.Linear(1200, self.latent_dim*2)

        self.lind1 = nn.Linear(latent_dim, 1200)
        self.lind2 = nn.Linear(1200, 1200)
        self.lind3 =nn.Linear(1200, 1200)
        self.lind4 = nn.Linear(1200, 4096)


    def encode(self, input):
        batch_size = input.size(0)
        input = input.view((batch_size, -1))
        result = torch.relu(self.line1(input))
        result = torch.relu(self.line2(result))
        mu_logvar = self.mu_logvar_gen(result)
        mu, logvar = mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

    def decode(self, input):

        batch_size = input.size(0)
        x = torch.tanh(self.lind1(input))
        x = torch.tanh(self.lind2(x))
        x = torch.tanh(self.lind3(x))
        x = torch.sigmoid(self.lind4(x))# Sigmoid because the distribution over pixels is supposed to be Bernoulli
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

        recon_loss =F.binary_cross_entropy(recon, x.view(batch_size, 4096), reduction='sum')
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