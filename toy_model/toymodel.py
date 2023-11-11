import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import random_sparse_input


class ToyModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=2, device="cpu"): 
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.rand(size=(hidden_dim, input_dim), device=device))
        self.relu1 = nn.ReLU()

    def encode(self, x):
        # x = torch.matmul(self.W, x)
        # make this batched
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.einsum("ij,bj->bi", self.W, x)
        return x

    def decode(self, x):
        # x = torch.matmul(self.W.T, x)
        # make this batched
        x = torch.einsum("ij,bj->bi", self.W.T, x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.relu1(x)
        x = self.decode(x)
        return x


def train_toymodel(input_dim, hidden_dim, train_cfg, device="cpu", use_wandb=False):
    model = ToyModel(input_dim, hidden_dim).to(device)
    num_iterations = train_cfg["num_iterations"]
    feature_importance = train_cfg["feature_importance"]
    feature_sparsity = train_cfg["feature_sparsity"]
    learning_rate = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]

    if feature_importance is None:
        feature_importance = torch.ones(input_dim, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(num_iterations):

        x = random_sparse_input(input_dim, feature_sparsity, batch_size=batch_size, device=device)

        y = model.forward(x)
        loss = torch.mean(feature_importance * (y - x) ** 2) 
        if use_wandb:
            wandb.log({"loss": loss.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"[{i}/{num_iterations}] loss: {loss.item()/batch_size/input_dim}")

    
    return model

def visualize(model):
    # plot the parameters
    params = model.state_dict()
    print(params)
    W = params["W"].cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.matmul(W.T, W), cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('W^T * W')
    plt.show()

def evaluate(model):
    # evaluate the model
    num_samples = 10000
    input_dim = model.input_dim
    # x = torch.rand(size=(num_samples, input_dim), device=device)
    x = random_sparse_input(input_dim, feature_sparsity=0.999)    
    y = model.forward(x)
    loss = torch.mean((y - x) ** 2)
    # normalize the loss over batch
    print(f"loss: {loss.item()}")

