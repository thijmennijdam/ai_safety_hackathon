import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import wandb 

from toymodel import ToyModel, train_toymodel, visualize
from sae import SAE, train_sae
from toy_sae_model import ToySAEModel

from utils import plot_inputs_and_outputs, plot_inputs_and_sae_encoding 
from tests import test_reconstruction


def main():

    use_wandb = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 30 
    hidden_dim = 20
 
    # importance_decay = 0.7
    # feature_importance = torch.from_numpy(np.array([importance_decay**i for i in range(input_dim)])).float().to(device)
    # feature_importance = torch.from_numpy(np.array([1, 1, 0, 0, 0])).float().to("cuda")
    feature_importance = torch.ones(input_dim, device=device)  # give equal importance to all features 

    train_toymodel_cfg = {
        "num_iterations": 10000,
        "learning_rate": 1e-3,
        "feature_importance": feature_importance, 
        "feature_sparsity": 0.8,  # feature sparsity is the probability of a feature being zero
        "batch_size": 32,
    }

    if use_wandb:
        wandb.init(project="toymodel", name="toymodel_{}_{}".format(input_dim, hidden_dim))


    # do_train_toymodel = True 
    do_train_toymodel = False
    
    if do_train_toymodel:
        # train the toy model
        toymodel = train_toymodel(input_dim=input_dim, hidden_dim=hidden_dim, train_cfg=train_toymodel_cfg)
        torch.save(toymodel.state_dict(), "toymodel_{}_{}.pt".format(input_dim, hidden_dim))
        # evaluate(toymodel)
        visualize(toymodel)
    else:
        # load the pretrained model
        toymodel = ToyModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        toymodel.load_state_dict(torch.load("toymodel_{}_{}.pt".format(input_dim, hidden_dim)))
        # visualize(toymodel)
    
    # plot_inputs_and_outputs(input_dim=input_dim, toymodel=toymodel)    

    # train SAE
    train_sae_cfg = {
        "num_iterations": 30000,
        "alpha_sparsity": .00001,
        "learning_rate": 1e-4, 
        "batch_size": 256,
    }
    
    # do_train_sae = True
    do_train_sae = False

    if do_train_sae:
        sae = train_sae(toymodel, input_dim=hidden_dim, feature_dim=input_dim, train_cfg=train_sae_cfg)
        torch.save(sae.state_dict(), "sae_{}_{}.pt".format(hidden_dim, input_dim))
    else:
        sae = SAE(input_dim=hidden_dim, feature_dim=input_dim).to(device)
        sae.load_state_dict(torch.load("sae_{}_{}.pt".format(hidden_dim, input_dim)))
    # plot_inputs_and_sae_encoding(sae, toymodel)


    # check if the reconstruction error is small
    toysaemodel = ToySAEModel(input_dim=input_dim, hidden_dim=hidden_dim, toymodel=toymodel, sae=sae, insert_sae=False)
    mse, stdev = test_reconstruction(toysaemodel)
    print(f"toymodel mse: {mse} ± {stdev}")
    toysaemodel = ToySAEModel(input_dim=input_dim, hidden_dim=hidden_dim, toymodel=toymodel, sae=sae, insert_sae=True)
    mse, stdev = test_reconstruction(toysaemodel)
    print(f"toysaemodel mse: {mse} ± {stdev}")

    
    # metric to gradient of weight, reconstruction loss to part of the features see which features are implicated

    # implement reconstruction loss with respect for one feature 
    # run the toysaemodel and backwards through the sae and see which features are implicated
    # by printing the gradients


    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
