import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict
import numpy as np
from sklearn import decomposition
from sklearn import linear_model

from tqdm import trange, tqdm
import torch
from torch.nn import functional as F

import wandb

class Evaluator():

    def __init__(self, model, device=torch.device("cpu"), sample_size=64,
                 dataset_size=1000, all_latents = False, use_wandb = True):
        self.sample_size = sample_size
        self.use_wandb = use_wandb
        self.dataset_size = dataset_size
        self.device = device
        self.all_latents = all_latents
        self.model = model.to(self.device)

    def __call__(self, data_loader):

        self.model.eval()

        return self.compute_metrics(data_loader)

    def compute_metrics(self, dataloader):
        

        accuracies = self._disentanglement_metric(dataloader.dataset, ["VAE", "PCA", "ICA"], sample_size=self.sample_size, dataset_size=self.dataset_size)
        
        if self.use_wandb:
            wandb.save("disentanglement_netrics.h5")
        print("accuracy:", accuracies)

        return accuracies

    def _disentanglement_metric(self, dataset, method_names, sample_size, n_epochs=6000, dataset_size = 1000, hidden_dim = 256, use_non_linear = False):

        #train models for all concerned methods and stor them in a dict
        methods = {}
        runtimes = {}
        for method_name in tqdm(method_names, desc="Iterating over methods for the Higgins disentanglement metric"):
            if method_name == "VAE":
                methods["VAE"] = self.model

            elif method_name == "PCA":   
                print("Training PCA...")
                pca = decomposition.PCA(n_components=self.model.latent_dim, whiten = True)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,0]
                    imgs_pca = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                else: 
                    imgs_pca = np.reshape(dataset.imgs, (dataset.imgs.shape[0], dataset.imgs.shape[1]**2))
                size = min(25000, len(imgs_pca))

                idx = np.random.randint(len(imgs_pca), size = size)
                imgs_pca = imgs_pca[idx, :]       #not enough memory for full dataset -> repeat with random subsets               
                pca.fit(imgs_pca)
                methods["PCA"] = pca
                print("Done")
                    

            elif method_name == "ICA":
                print("Training ICA...")
                ica = decomposition.FastICA(n_components=self.model.latent_dim, max_iter=400)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,0]
                    imgs_ica = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                else:
                    imgs_ica = np.reshape(dataset.imgs, (dataset.imgs.shape[0], dataset.imgs.shape[1]**2))
                size = min(1000, len(imgs_ica))
                idx = np.random.randint(len(imgs_ica), size = size)
                imgs_ica = imgs_ica[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                ica.fit(imgs_ica)
                methods["ICA"] = ica
                print("Done")

            else: 
                raise ValueError("Unknown method : {}".format(method_name))
         

        data_train, data_test = {}, {}

        for method in methods:
            data_train[method] = [], []
            data_test[method] = [], []

        #latent dim = length of z_b_diff for arbitrary method = output dimension of linear classifier
        latent_dim = 10

        #generate dataset_size many training data points and 20% of that test data points
        for i in tqdm(range(dataset_size), desc="Generating datasets for Higgins metric"):
            data = self._compute_z_b_diff_y(methods, sample_size, dataset)
            for method in methods:
                data_train[method][0].append(data[method][0])
                data_train[method][1].append(data[method][1])
            if i <= int(dataset_size*0.5):
                
                data = self._compute_z_b_diff_y(methods, sample_size, dataset)
                for method in methods:
                    data_test[method][0].append(data[method][0])
                    data_test[method][1].append(data[method][1])

        test_acc = {"linear":{}}


        for method in tqdm(methods.keys(), desc = "Training classifiers for the Higgins metric"):
            classifier = linear_model.LogisticRegression(max_iter=500)
            X_train, Y_train = data_train[method]
            X_test, Y_test = data_test[method]
            classifier.fit(X_train, Y_train)
            train_acc = np.mean(classifier.predict(X_train)==Y_train)
            test_acc["linear"][method] = np.mean(classifier.predict(X_test)==Y_test)
            print(f'Accuracy of {method} on training set: {train_acc:.4f}, test set: {test_acc["linear"][method].item():.4f}')

        return test_acc


    def _compute_z_b_diff_y(self, methods, sample_size, dataset):
        """
        Compute the disentanglement metric score as proposed in the original paper
        reference: https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
        """
        #if dsprites:

        if dataset.lat_sizes.size == 5 and not self.all_latents:
            
            y = np.random.randint(1, dataset.lat_sizes.size, size=1)
        else:
            y = np.random.randint(dataset.lat_sizes.size, size=1)
        
        
        #y = np.random.randint(dataset.lat_sizes.size, size=1)
        y_lat = np.random.randint(dataset.lat_sizes[y], size=sample_size)

        # Helper function to show images
        def show_images_grid(imgs_, num_images=25):
            ncols = int(np.ceil(num_images**0.5))
            nrows = int(np.ceil(num_images / ncols))
            _, axes = plt.pyplot.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
            axes = axes.flatten()

            for ax_i, ax in enumerate(axes):
                if ax_i < num_images:
                    ax.imshow(imgs_[ax_i][0], cmap='Greys_r',  interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
            plt.pyplot.show()
  
        #print("y", y)
        #print("ylat", y_lat)
        imgs_sampled1  = dataset.images_from_data_gen(sample_size, y, y_lat)
        imgs_sampled2  = dataset.images_from_data_gen(sample_size, y, y_lat)
        #show_images_grid(imgs_sampled1)
        #show_images_grid(imgs_sampled2)
        res = {}
        #calculate the expectation values of the normal distributions in the latent representation for the given images
        for method in methods.keys():
            if method == "VAE":
                with torch.no_grad():
                    mu1_torch, _ = self.model.encode(imgs_sampled1.to(self.device))
                    mu1 = mu1_torch.cpu().detach().numpy()
                    mu2_torch, _ = self.model.encode(imgs_sampled2.to(self.device))
                    mu2 = mu2_torch.cpu().detach().numpy()  
            elif method == "PCA":
                pca = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1[:,0,:,:], (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2[:,0,:,:], (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                else:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                mu1 = pca.transform(imgs_sampled_pca1)
                mu2 = pca.transform(imgs_sampled_pca2)

            elif method == "ICA":
                ica = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_ica1 = torch.reshape(imgs_sampled1[:,0,:,:], (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_ica2 = torch.reshape(imgs_sampled2[:,0,:,:], (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                else:   
                   imgs_sampled_ica1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                   imgs_sampled_ica2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                mu1 = ica.transform(imgs_sampled_ica1)
                mu2 = ica.transform(imgs_sampled_ica2)
            else: 
                raise ValueError("Unknown method : {}".format(method)) 

            res[method] = np.mean(np.abs(mu1 - mu2), axis=0), y[0]

        return res