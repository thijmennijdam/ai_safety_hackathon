import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import wandb 

from toymodel import ToyModel, train_toymodel, visualize
from utils import plot_inputs_and_outputs, plot_inputs_and_sae_encoding
from sae import SAE, train_sae


def main():

    use_wandb = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 6 
    hidden_dim = 4 
 
    # importance_decay = 0.7
    # feature_importance = torch.from_numpy(np.array([importance_decay**i for i in range(input_dim)])).float().to(device)
    # feature_importance = torch.from_numpy(np.array([1, 1, 0, 0, 0])).float().to("cuda")
    
    train_cfg = {
        "num_iterations": 100000,
        "learning_rate": 1e-5,
        "feature_importance": torch.ones(input_dim, device=device),  # give equal importance to all features
        "feature_sparsity": 0.8,  # feature sparsity is the probability of a feature being zero
        "batch_size": 32,
    }

    if use_wandb:
        wandb.init(project="toymodel", name="feature_importance_0.8") 


    # train_toymodel = True 
    train_toymodel = False
    
    if train_toymodel:
        # train the toy model
        toymodel = train_toymodel(input_dim=input_dim, hidden_dim=hidden_dim, train_cfg=train_cfg)
        visualize(toymodel)
        torch.save(toymodel.state_dict(), "toymodel_{}_{}.pt".format(input_dim, hidden_dim))
        # evaluate(toymodel)
    else:
        # load the pretrained model
        toymodel = ToyModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        toymodel.load_state_dict(torch.load("toymodel_{}_{}.pt".format(input_dim, hidden_dim)))
    

    # train SAE
    train_cfg = {
        "num_iterations": 10000,
        "alpha_sparsity": .1,
        "learning_rate": 1e-2, 
    }
    
    # bool_train_sae = True
    bool_train_sae = False
    if bool_train_sae:
        sae = train_sae(toymodel, input_dim=hidden_dim, feature_dim=input_dim, train_cfg=train_cfg)
        torch.save(sae.state_dict(), "sae_{}_{}.pt".format(hidden_dim, input_dim))
    else:
        sae = SAE(input_dim=hidden_dim, feature_dim=input_dim).to(device)
        sae.load_state_dict(torch.load("sae_{}_{}.pt".format(hidden_dim, input_dim)))

    if use_wandb:
        wandb.finish()
    # test the pretrained model
    # x = torch.rand(size=(input_dim,), device=device)
    # plot_inputs_and_outputs(input_dim=input_dim, toymodel=toymodel)    
    plot_inputs_and_sae_encoding(sae, toymodel)

if __name__ == "__main__":
    main()
    # test_random_sparse_input()
