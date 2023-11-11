import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.W_enc = nn.Parameter(torch.rand(size=(feature_dim, input_dim)))
        self.bias_enc = nn.Parameter(torch.rand(size=(feature_dim,)))
        self.W_dec = nn.Parameter(torch.rand(size=(input_dim, feature_dim)))
        self.bias_dec = nn.Parameter(torch.rand(size=(input_dim,)))
    
    def encode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # return torch.relu(self.W_enc @ x + self.bias_enc)
        print(self.W_enc.shape, x.shape)
        return torch.relu(torch.einsum("ij,bj->bi", self.W_enc, x) )#+ self.bias_enc)

    def decode(self, x):
        # return self.W_dec @ x + self.bias_dec
        return torch.einsum("ij,bj->bi", self.W_dec, x) + self.bias_dec

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


    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    for param in toymodel.parameters():
        param.requires_grad = False
    
    for i in range(num_iterations):

        x = torch.rand(size=(toymodel.input_dim,), device=device)
        h = toymodel.encode(x)

        reconstruction_loss = torch.mean((sae.forward(h) - h) ** 2) 
        
        # the sparsity loss is the L1 norm of the encoded vector 
        sparsity_loss = torch.sum(torch.abs(sae.encode(h)))
        loss = reconstruction_loss + alpha_sparsity * sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"[{i}/{num_iterations}] sae-loss: {loss.item():.3f}")
            
    return sae

