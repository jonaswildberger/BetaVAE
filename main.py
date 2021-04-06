import yaml
import argparse
import numpy as np
from torch import optim
import random
import sys


from models.betaVAEHiggins import *
from models.betaVAEConv import *
from models.betaVAEBurgess import *
from datasets import *
from training import Trainer
from evaluate import Evaluator

import wandb

CONFIG_FILE = "hyperparam.ini"


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-name', '--name', type=str,
                         help="Name of the model for storing and loading purposes.")
    
    
    general.add_argument('-s', '--seed', type=int, default=None,
                         help='Random seed. Can be `None` for stochastic behavior.')
   
    # Learning options
    training = parser.add_argument_group('Training specific options')
    
    training.add_argument('-e', '--epochs', type=int,
                          default=10,
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=256,
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=0.01,
                          help='Learning rate.')

    training.add_argument('--dry_run', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,
                        help='Whether to use WANDB in offline mode.')
    training.add_argument('--wandb_log', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,
                help='Whether to use WANDB - this has implications for the training loop since if we want to log, we compute the metrics over training')                
    training.add_argument('--wandb_key', type=str, default=None,
                help='Path to WANDB key')    
   
    training.add_argument('--sample_size', type=int, default=64,
        help='Whether to drop UMAP/TSNE etc. for computing Higgins metric (if we do not drop them, generating the data takes ~25 hours)')      
    training.add_argument('--dataset_size', type=int, default=1000,
        help='Whether to drop UMAP/TSNE etc. for computing Higgins metric (if we do not drop them, generating the data takes ~25 hours)')      
    training.add_argument('--all_latents', type=lambda x: False if x in ["False", "false", "", "None", "0"] else True, default=0,
        help='Whether to use 5 or 4 latents in Dsprites')      
    training.add_argument('--loss-b', type=float, default=1.,
        help='beta factor for loss')

    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default='BetaVAEHiggins', choices=['BetaVAEHiggins', 'BetaVAEConv', 'BetaVAEBurgess'],
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                      default = 10,
                       help='Dimension of the latent variable.')

    args = parser.parse_args(args_to_parse)

    return args



def main(args):


    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    try:
        wandb.login()
    except:
        print(f"Authentication for WANDB failed! Trying to disable it")
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project='atmlbetavae', entity='atml', group="jonas")
    wandb.config.update(args)
    

    args.seed = args.seed if args.seed is not None else random.randint(1,10000)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    print("Config:")
    print(vars(args))


    train_loader, test_loader = get_dataloaders("dsprites",
                                       batch_size=args.batch_size, shuffle=True)
    img_size = train_loader.dataset.img_size
    beta = args.loss_b
    print(beta)
    model = eval("{model}({latent_dim}, {beta}, {img_size}, latent_dist = 'bernoulli')".format(model = args.model_type, latent_dim = args.latent_dim, beta = beta, img_size = img_size))
    print(model)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    

    trainer = Trainer(model, optimizer, device = device)
    print("training")
    trainer(train_loader, epochs=args.epochs)

                                                                                                                
    evaluator = Evaluator(model, device=device, sample_size = args.sample_size, dataset_size = args.dataset_size, all_latents= args.all_latents, use_wandb = args.wandb_log)

    metrics = evaluator(test_loader)
    wandb.log({"final":{"metric":metrics}})
   
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)