import torch
from torch import nn
from utils import random_sparse_input


class SAE(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        # orthogonal initialization
        W_enc_orthogonal = torch.nn.init.orthogonal_(torch.empty(size=(feature_dim, input_dim)))
        # self.W_enc = nn.Parameter(torch.rand(size=(feature_dim, input_dim)))
        self.W_enc = nn.Parameter(W_enc_orthogonal)
        self.bias_enc = nn.Parameter(torch.rand(size=(feature_dim,)))
        self.W_dec = nn.Parameter(W_enc_orthogonal.T)
        # self.bias_dec = nn.Parameter(torch.rand(size=(input_dim,))) → Sharkey et al.: no bias
    
    def encode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return torch.relu(torch.einsum("ij,bj->bi", self.W_enc, x) + self.bias_enc)

    def decode(self, x):
        return torch.einsum("ij,bj->bi", self.W_dec, x) 

    def forward(self, x):
        return self.decode(self.encode(x))
    

def train_sae(toymodel, input_dim, feature_dim, train_cfg, device="cpu"):
    """
    Train the sparse autoencoder on the hidden layer of the pretrained model.
    """
    sae = SAE(input_dim=input_dim, feature_dim=feature_dim).to(device)

    alpha_sparsity = train_cfg["alpha_sparsity"]
    num_iterations = train_cfg["num_iterations"]
    learning_rate = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]


    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    for param in toymodel.parameters():
        param.requires_grad = False
    
    for i in range(num_iterations):

        # x = torch.rand(size=(toymodel.input_dim,), device=device)
        # print('random_x')
        # print(x)
        # h = toymodel.encode(x)
        # print('random h')
        # print(h)
        x = random_sparse_input(toymodel.input_dim, feature_sparsity=0.8, batch_size=batch_size, device=device)
        # print('random_sparse_x')
        # print(x)
        h = toymodel.encode(x)
        # print('random h')
        # print(h)
        # print('h')
        # print(h.shape)

        reconstruction_loss = torch.mean((sae.forward(h) - h) ** 2)  / sae.input_dim 
        
        # the sparsity loss is the L1 norm of the encoded vector 
        sparsity_loss = torch.sum(torch.abs(sae.encode(h))) / sae.feature_dim
        loss = reconstruction_loss + alpha_sparsity * sparsity_loss 
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"[{i}/{num_iterations}] sae-loss: {loss.item()}")
            
    return sae

