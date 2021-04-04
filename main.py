import yaml
import argparse
import numpy as np
from torch import optim
import random


from models.betaVAE import *
from datasets import *
from training import Trainer
from evaluate import Evaluator





def main():

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


    train_loader = get_dataloaders("dsprites",
                                       batch_size=256, shuffle=True)
    model = BetaVAE(10, beta=4) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    

    trainer = Trainer(model, optimizer, device = device)
    print("training")
    trainer(train_loader, epochs=1)


    test_loader = get_dataloaders('dsprites',
                                      batch_size=256, shuffle = False)

    evaluator = Evaluator(model, device=device)

    evaluator(test_loader)
   
if __name__ == '__main__':
    main()