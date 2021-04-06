import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np

class BetaVAEBurgess(BaseVAE):
    model_type = "BetaVAEBurgess"
    num_iter = 0
    def __init__(self,
                latent_dim,
                beta = 1,
                img_size = (1, 64, 64),
                 latent_dist = 'bernoulli'):

        super(BetaVAEBurgess, self).__init__()
        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.beta = beta
        self.latent_dist = latent_dist
        self.img_size = img_size
        
        input_channels = self.img_size[0]
        hidden_channels = 32
        kernel_size = 4
        stride = 2
        padding = 1
        hidden_dim = 256

        self.reshape = (hidden_channels, kernel_size, kernel_size)

        self.conve1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, stride = stride, padding= padding)
        self.conve2 =  nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding)
        self.conve3 =  nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding)

        if self.img_size[1] == 64:
            self.conve4 =  nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding)

        self.line1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.line2 = nn.Linear(hidden_dim, hidden_dim) 
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim*2)


        self.lind1 = nn.Linear(latent_dim, hidden_dim)
        self.lind2 = nn.Linear(hidden_dim, hidden_dim)
        self.lind3 = nn.Linear(hidden_dim, np.product(self.reshape))
        self.convd1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding)
        self.convd2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding) 
        self.convd3 = nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size, stride = stride, padding= padding)
        if self.img_size[1] == 64:
            self.convd4 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, stride = stride, padding= padding) 
        



    def encode(self, input):
        
        batch_size = input.size(0)

        x = torch.relu(self.conve1(input))
        x = torch.relu(self.conve2(x))
        x = torch.relu(self.conve3(x))
        if self.img_size[1] == 64: 
            x = torch.relu(self.conve4(x))

        x = x.view((batch_size, -1))
        x = torch.relu(self.line1(x))
        x = torch.relu(self.line2(x))

        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

    def decode(self, input):

        batch_size = input.size(0)

        x = torch.relu(self.lind1(input))
        x = torch.relu(self.lind2(x))
        x = torch.relu(self.lind3(x))
        x = x.view(batch_size, *self.reshape)

        x = torch.relu(self.convd1(x))
        x = torch.relu(self.convd2(x))
        if self.img_size[1] == 64:
            x = torch.relu(self.convd4(x))
        x = torch.sigmoid(self.convd3(x))

       
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon, x, mu, log_var, storer = None):
        self.num_iter += 1
        batch_size = x.size(0)
        if self.latent_dist == 'bernoulli':
            recon_loss =F.binary_cross_entropy(recon, x, reduction='sum')
        elif self.latent_dist  == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
            recon_loss = F.mse_loss(recon * 255, x * 255, reduction="sum") / 255
        #recon_loss/= batch_size
        latent_kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=0)
        kld_loss = torch.sum(latent_kl)

        
        loss = recon_loss + self.beta  * kld_loss

        if storer is not None:
            storer['recon_loss'].append(recon_loss.item())
            storer['kl_loss'].append(kld_loss.item())
            for i in range(self.latent_dim):
                
                storer['kl_loss_' + str(i)].append(latent_kl[i].item())
            storer['loss'].append(loss.item())

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

       
