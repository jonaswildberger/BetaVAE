import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict
import numpy as np
from sklearn import manifold
import sklearn.ensemble
from sklearn import decomposition
from sklearn import linear_model
from models.classifier import Classifier, weight_reset

from tqdm import trange, tqdm
import torch
from torch.nn import functional as F
from utils.fid import get_fid_value
import time 
import wandb

import matplotlib.pyplot as plt

class Evaluator():

    def __init__(self, model, device=torch.device("cpu"), sample_size=64,
                 dataset_size=1000,  logger=logging.getLogger(__name__), all_latents = False, use_NN_classifier = False, use_wandb = True, seed = None, higgins_drop_slow = True, multiple_l=False):
        self.sample_size = sample_size
        self.use_wandb = use_wandb
        self.dataset_size = dataset_size
        self.device = device
        self.all_latents = all_latents
        self.use_NN_classifier = False
        self.model = model.to(self.device)
        self.seed = seed
        self.logger = logger
        self.higgins_drop_slow = higgins_drop_slow
        self.multiple_l = multiple_l

    def __call__(self, data_loader, dataset_name = None):
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()


        self.logger.info('Computing losses...')
        losses = self.compute_losses(data_loader)
        self.logger.info('Losses: {}'.format(losses))

        self.logger.info('Computing metrics...')
        metrics = self.compute_metrics(data_loader, dataset_name)
        self.logger.info('Metrics: {}'.format(metrics))

        if is_still_training:
            self.model.train()
        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return metrics


    def compute_losses(self, dataloader, batch_size=1):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        self.model.eval()
        storer = defaultdict(list)
        for data, _ in tqdm(dataloader, leave=False):
            data = data.to(self.device)

            recon_batch, mu, logvar = self.model(data)
            _ = self.model.loss_function(recon_batch, data, mu, logvar, storer=storer)
            
        losses = {k: sum(v) / len(dataloader)/batch_size for k, v in storer.items()}
        self.model.train()
        return losses

    def compute_metrics(self, dataloader, dataset_name = None):
        
        accuracies, aam, mig, fid = None, None, None, None
        #TODO: dont run if on collab
        
        if dataset_name != '3dshapes' and dataset_name != 'mpi3dtoy' and dataset_name != 'dsprites':
            fid = get_fid_value(dataloader, self.model, batch_size=dataloader.batch_size)
        
        if dataset_name in ['dsprites', 'mpi3dtoy', '3dshapes']: 
            self.logger.info("Computing the disentanglement metric")
            method_names = ["VAE", "PCA", "ICA", "T-SNE","UMAP", "DensUMAP"]
            if self.multiple_l is False:
                accuracies = self._disentanglement_metric(dataloader.dataset, method_names, sample_size=self.sample_size, n_epochs = 10000, dataset_size=self.dataset_size)
            else:
                Ls = [16,64,128,256]
                accuracies = {"L"+sample_size: self._disentanglement_metric(dataloader.dataset, method_names, sample_size=self.sample_size, 
                    n_epochs = 10000, dataset_size=self.dataset_size) for sample_size in Ls}

        if self.use_wandb:
                # wandb.log(accuracies)
            wandb.save("disentanglement_metrics.h5")
        
        if dataset_name in ['dsprites']:
            try:
                lat_sizes = dataloader.dataset.lat_sizes
                lat_names = dataloader.dataset.lat_names
                lat_imgs = dataloader.dataset.imgs
            except AttributeError:
                raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))
            
            self.logger.info("Computing the empirical distribution q(z|x).")
            samples_zCx, params_zCx = self._compute_q_zCx(dataloader)
            len_dataset, latent_dim = samples_zCx.shape

            self.logger.info("Estimating the marginal entropy.")
            # marginal entropy H(z_j)
            H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)

            # conditional entropy H(z|v)
            samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
            params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
            H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)

            H_z = H_z.cpu()
            H_zCv = H_zCv.cpu()

            # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
            mut_info = - H_zCv + H_z
            sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)

            metric_helpers = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
            mig = self._mutual_information_gap(sorted_mut_info, lat_sizes, storer=metric_helpers).item()
            aam = self._axis_aligned_metric(sorted_mut_info, storer=metric_helpers).item()
        

        metrics = {'DM': accuracies, 'MIG': mig, 'AAM': aam, 'FID': fid}
        print(f"Evaluated metrics for {dataset_name} as: {metrics}")
        self.model.train()
        return metrics

        
        

    def _disentanglement_metric(self, dataset, method_names, sample_size, n_epochs=6000, dataset_size = 1000, hidden_dim = 256, use_non_linear = False):

        #train models for all concerned methods and stor them in a dict
        methods = {}
        runtimes = {}
        for method_name in tqdm(method_names, desc="Iterating over methods for the Higgins disentanglement metric"):
            if method_name == "VAE":
                methods["VAE"] = self.model

            elif method_name == "PCA":   
                start = time.time()
                print("Training PCA...")
                pca = decomposition.PCA(n_components=self.model.latent_dim, whiten = True, random_state=self.seed)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,:]
                    print(f"Shape of data images: {data_imgs.shape}")
                    imgs_pca = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[3]*data_imgs.shape[1]**2))
                else:
                    data_imgs = dataset.imgs 
                    imgs_pca = np.reshape(dataset.imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                size = min(3500 if (len(data_imgs.shape) > 3 and data_imgs.shape[3]) > 1 else 25000, len(imgs_pca))

                idx = np.random.randint(len(imgs_pca), size = size)
                imgs_pca = imgs_pca[idx, :]       #not enough memory for full dataset -> repeat with random subsets               
                pca.fit(imgs_pca)
                methods["PCA"] = pca
                
                self.logger.info("Done")

                runtimes[method_name] = time.time()-start

            elif method_name == "ICA":
                start = time.time()
                print("Training ICA...")
                ica = decomposition.FastICA(n_components=self.model.latent_dim, max_iter=400, random_state=self.seed)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,:]
                    print(f"Shape of data images: {data_imgs.shape}")
                    imgs_ica = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[3]*data_imgs.shape[1]**2))
                else:
                    data_imgs = dataset.imgs 
                    imgs_ica = np.reshape(dataset.imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                size = min(1000 if (len(data_imgs.shape) > 3 and data_imgs.shape[3]) > 1 else 2500, len(imgs_ica))
                idx = np.random.randint(len(imgs_ica), size = size)
                imgs_ica = imgs_ica[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                ica.fit(imgs_ica)
                methods["ICA"] = ica
                
                self.logger.info("Done")

                runtimes[method_name] = time.time()-start

            elif method_name == "T-SNE":
                continue
               
            elif method_name == "UMAP":
                if self.higgins_drop_slow:
                    continue
                else:
                    start = time.time() 
                    import umap
                    self.logger.info("Training UMAP...")
                    umap_model = umap.UMAP(random_state=self.seed, densmap=False, n_components=self.model.latent_dim)
                    imgs_umap = np.reshape(dataset.imgs, (dataset.imgs.shape[0], dataset.imgs.shape[1]**2))
                    size = min(25000, len(imgs_umap))
                    idx = np.random.randint(len(imgs_umap), size = size)
                    imgs_umap = imgs_umap[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                    umap_model.fit(imgs_umap)
                    methods["UMAP"] = umap_model
                    self.logger.info("Done")
                    runtimes[method_name] = time.time()-start

            elif method_name == "DensUMAP":
                continue

            else: 
                raise ValueError("Unknown method : {}".format(method_name))
         
        if self.use_wandb:
            try:
                wandb.log(runtimes)
            except:
                pass

        data_train, data_test = {}, {}

        for method in methods:
            data_train[method] = [], []
            data_test[method] = [], []

        #latent dim = length of z_b_diff for arbitrary method = output dimension of linear classifier
        latent_dim = self.model.latent_dim

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


        test_acc = {"logreg":{},"linear":{}, "nonlinear":{}, "rf":{}}
        for model_class in ["linear", "nonlinear", "logreg", "rf"]:
            if model_class in ["linear", "nonlinear"]:
                model = Classifier(latent_dim,hidden_dim,len(dataset.lat_sizes), use_non_linear= True if model_class =="nonlinear" else False)
                
                model.to(self.device)
                model.train()

                #log softmax with NLL loss 
                criterion = torch.nn.NLLLoss()
                optim = torch.optim.Adagrad(model.parameters(), lr=0.01 if model_class =="linear" else 0.001, weight_decay=0 if model_class == "linear" else 1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5000, min_lr=0.00001)

                for method in tqdm(methods.keys(), desc = "Training classifiers for the Higgins metric"):
                    if method == "ICA":
                        optim = torch.optim.Adam(model.parameters(), lr=1 if model_class =="linear" else 0.001, weight_decay=0 if model_class == "linear" else 1e-4)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5000, min_lr=0.00001)
                    X_train, Y_train = data_train[method]
                    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
                    X_train = X_train.to(self.device)
                    Y_train = Y_train.to(self.device)
                    
                    X_test , Y_test = data_test[method]
                    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
                    X_test = X_test.to(self.device)
                    Y_test = Y_test.to(self.device)

                    print(f'Training the classifier for model {method}')
                    for e in tqdm(range(n_epochs if model_class == "linear" else round(n_epochs/2)), desc="Iterating over epochs while training the Higgins classifier"):
                        model.train()
                        optim.zero_grad()
                        
                        scores_train = model(X_train)   
                        loss = criterion(scores_train, Y_train)
                        loss.backward()
                        optim.step()
                        scheduler.step(loss)
                        
                        if (e+1) % 2000 == 0:
                            model.eval()
                            with torch.no_grad():
                                scores_test = model(X_test)   
                                test_loss = criterion(scores_test, Y_test)
                                tqdm.write(f'In this epoch {e+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {test_loss.item():.4f}')
                                model.eval()
                                scores_train = model(X_train)
                                scores_test = model(X_test)
                                _, prediction_train = scores_train.max(1)
                                _, prediction_test = scores_test.max(1)

                                train_acc = (prediction_train==Y_train).sum().float()/len(X_train)
                                test_acc[model_class][method] = (prediction_test==Y_test).sum().float()/len(X_test)
                                tqdm.write(f'Accuracy of {method} on training set: {train_acc.item():.4f}, test set: {test_acc[model_class][method].item():.4f}')
                            model.train()
                    
                    model.eval()
                    with torch.no_grad():
                        
                        scores_train = model(X_train)
                        scores_test = model(X_test)
                        _, prediction_train = scores_train.max(1)
                        _, prediction_test = scores_test.max(1)

                        train_acc = (prediction_train==Y_train).sum().float()/len(X_train)
                        test_acc[model_class][method] = (prediction_test==Y_test).sum().float()/len(X_test)
                        print(f'Accuracy of {method} on training set: {train_acc.item():.4f}, test set: {test_acc[model_class][method].item():.4f}')
                        
                    model.apply(weight_reset)
                
            elif model_class in ["logreg", "rf"]:

                for method in tqdm(methods.keys(), desc = "Training classifiers for the Higgins metric"):
                    if model_class == "logreg":
                        classifier = linear_model.LogisticRegression(max_iter=500, random_state=self.seed)
                    elif model_class == "rf":
                        classifier = sklearn.ensemble.RandomForestClassifier(n_estimators = 150)
                    X_train, Y_train = data_train[method]
                    X_test, Y_test = data_test[method]
                    classifier.fit(X_train, Y_train)
                    train_acc = np.mean(classifier.predict(X_train)==Y_train)
                    test_acc[model_class][method] = np.mean(classifier.predict(X_test)==Y_test)
                    print(f'Accuracy of {method} on training set: {train_acc:.4f}, test set: {test_acc[model_class][method].item():.4f}')

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
        
        
        y_lat = np.random.randint(dataset.lat_sizes[y], size=sample_size)

        # Helper function to show images
        def show_images_grid(imgs_, num_images=25):
            ncols = int(np.ceil(num_images**0.5))
            nrows = int(np.ceil(num_images / ncols))
            _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
            axes = axes.flatten()

            for ax_i, ax in enumerate(axes):
                if ax_i < num_images:
                    ax.imshow(imgs_[ax_i][0], cmap='Greys_r',  interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
            plt.show()
  

        imgs_sampled1  = dataset.images_from_data_gen(sample_size, y, y_lat)
        imgs_sampled2  = dataset.images_from_data_gen(sample_size, y, y_lat)

        res = {}
        #calculate the expectation values of the normal distributions in the latent representation for the given images
        for method in methods.keys():
            if method == "VAE":
                with torch.no_grad():
                    mu1, _ = self.model.encode(imgs_sampled1.to(self.device))
                    mu2, _ = self.model.encode(imgs_sampled2.to(self.device))  
                    if not self.use_NN_classifier:
                        mu1 = mu1.cpu().detach().numpy()
                        mu2 = mu2.cpu().detach().numpy()  
            elif method == "PCA":
                pca = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[1]*imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[1]*imgs_sampled2.shape[2]**2))
                else:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                mu1 = pca.transform(imgs_sampled_pca1)
                mu2 = pca.transform(imgs_sampled_pca2)

            elif method == "ICA":
                ica = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_ica1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[1]*imgs_sampled1.shape[2]**2))
                    imgs_sampled_ica2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[1]*imgs_sampled2.shape[2]**2))
                else:   
                   imgs_sampled_ica1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                   imgs_sampled_ica2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                mu1 = ica.transform(imgs_sampled_ica1)
                mu2 = ica.transform(imgs_sampled_ica2)
            elif method == "T-SNE":
                continue
                # tsne = methods[method]
                
                # #flatten images
                # imgs_sampled_tsne1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                # imgs_sampled_tsne2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                # mu1 = torch.from_numpy(tsne.fit_transform(imgs_sampled_tsne1)).float()
                # mu2 = torch.from_numpy(tsne.fit_transform(imgs_sampled_tsne2)).float()
            elif method == "UMAP":
                if self.higgins_drop_slow:
                    continue
                else:
                    umap = methods[method]
                    #flatten images
                    imgs_sampled1 = imgs_sampled1[0:100]
                    imgs_sampled2 = imgs_sampled2[0:100]
                    if dataset.imgs.ndim == 4:
                        imgs_sampled_umap1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[1]*imgs_sampled1.shape[2]**2))
                        imgs_sampled_umap2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[1]*imgs_sampled2.shape[2]**2))
                    else:   
                        imgs_sampled_umap1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                        imgs_sampled_umap2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                    if not self.use_NN_classifier:
                        mu1 = umap.transform(imgs_sampled_umap1)
                        mu2 = umap.transform(imgs_sampled_umap2)
                    else:
                        mu1 = torch.from_numpy(umap.transform(imgs_sampled_umap1)).float()
                        mu2 = torch.from_numpy(umap.transform(imgs_sampled_umap2)).float()
            elif method == "DensUMAP":
                continue
                # densumap = methods[method]
                # #flatten images
                # imgs_sampled_densumap1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                # imgs_sampled_densumap2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                # mu1 = torch.from_numpy(densumap.fit_transform(imgs_sampled_densumap1)).float()
                # mu2 = torch.from_numpy(densumap.fit_transform(imgs_sampled_densumap2)).float()
                
            else: 
                raise ValueError("Unknown method : {}".format(method)) 
            res[method] = np.mean(np.abs(mu1 - mu2), axis=0), y[0]

        return res



    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of variation
        # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _axis_aligned_metric(self, sorted_mut_info, storer=None):
        """Compute the proposed axis aligned metrics."""
        numerator = (sorted_mut_info[:, 0] - sorted_mut_info[:, 1:].sum(dim=1)).clamp(min=0)
        aam_k = numerator / sorted_mut_info[:, 0]
        aam_k[torch.isnan(aam_k)] = 0
        aam = aam_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["aam_k"] = aam_k
            storer["aam"] = aam

        return aam

    def _compute_q_zCx(self, dataloader):
        """Compute the empiricall disitribution of q(z|x).

        Parameter
        ---------
        dataloader: torch.utils.data.DataLoader
            Batch data iterator.

        Return
        ------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
        """
        len_dataset = len(dataloader.dataset)
        latent_dim = self.model.latent_dim
        n_suff_stat = 2

        q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device)

        n = 0
        with torch.no_grad():
            for x, label in dataloader:
                batch_size = x.size(0)
                idcs = slice(n, n + batch_size)
                q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = self.model.encoder(x.to(self.device))
                n += batch_size

        params_zCX = q_zCx.unbind(-1)
        samples_zCx = self.model.reparameterize(*params_zCX)

        return samples_zCx, params_zCX

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
        r"""Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
        using the emperical distribution of :math:`p(x)`.

        Note
        ----
        - the expectation over the emperical distributio is: :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
        - we assume that q(z|x) is factorial i.e. :math:`q(z|x) = \prod_j q(z_j|x)`.
        - computes numerically stable NLL: :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

        Parameters
        ----------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

        Return
        ------
        H_z: torch.Tensor
            Tensor of shape (latent_dim) containing the marginal entropies H(z_j)
        """
        len_dataset, latent_dim = samples_zCx.shape
        device = samples_zCx.device
        H_z = torch.zeros(latent_dim, device=device)

        # sample from p(x)
        samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
        # sample from p(z|x)
        samples_zCx = samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = params_zCX[0].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_var = params_zCX[1].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)
        with trange(n_samples, leave=False, disable=self.is_progress_bar) as t:
            for k in range(0, n_samples, mini_batch_size):
                # log q(z_j|x) for n_samples
                idcs = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                 mean[..., idcs],
                                                 log_var[..., idcs])
                # numerically stable log q(z_j) for n_samples:
                # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
                # As we don't know q(z) we appoximate it with the monte carlo
                # expectation of q(z_j|x_n) over x. => fix a single z and look at
                # proba for every x to generate it. n_samples is not used here !
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
                # H(z_j) = E_{z_j}[- log q(z_j)]
                # mean over n_samples (i.e. dimesnion 1 because already summed over 0).
                H_z += (-log_q_z).sum(1)

                t.update(mini_batch_size)

        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """Estimate conditional entropies :math:`H[z|v]`."""
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes)
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device)
        for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for i in range(lat_size):
                self.logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
                idcs[i_fac_var] = i
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size,
                                                                   latent_dim)
                params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                    for p in params_zCx)

                H_zCv[i_fac_var] += self._estimate_latent_entropies(samples_zxCv, params_zxCv
                                                                    ) / lat_size
        return H_zCv
