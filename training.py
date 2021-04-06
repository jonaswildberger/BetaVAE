import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F


class Trainer():

    def __init__(self, model, optimizer, device=torch.device("cpu")):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer


    def __call__(self, data_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):

            epoch_loss = 0
            kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=False)
            with trange(len(data_loader), **kwargs) as t:

                for _, (data, _) in enumerate(data_loader):
                    batch_size, _, _, _ = data.size()
                    
                    
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    recon_batch, mu, logvar = self.model(data)

                    loss = self.model.loss_function(recon_batch, data, mu,logvar)/len(data)

                    
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(loss=loss)
                    t.update()

            mean_epoch_loss = epoch_loss / len(data_loader)
            print('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,mean_epoch_loss))