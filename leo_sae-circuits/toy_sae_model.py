# construct the new model
# new model is the composition of the toy model and the SAE
# the SAE is inserted in the middle of the toy model 
# toy model: input_dim → hidden_dim
# SAE: hidden_dim → input_dim → hidden_dim
# toy model: hidden_dim → input_dim
# the new model: input_dim → hidden_dim → input_dim

# from sae import SAE
# from toymodel import ToyModel

import torch.nn as nn

class ToySAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, toymodel, sae, insert_sae=True):
        super().__init__()
        self.insert_sae = insert_sae
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.toymodel = toymodel
        self.sae = sae
    
    def encode(self, x):
        if self.insert_sae: 
            return self.sae.encode(self.toymodel.encode(x))
        else:
            return self.toymodel.encode(x)
    
    def decode(self, x):
        if self.insert_sae:
            return self.toymodel.decode(self.sae.decode(x))
        else:
            return self.toymodel.decode(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))