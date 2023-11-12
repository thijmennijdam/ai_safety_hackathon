import torch
import numpy as np
from utils import random_sparse_input

def test_sae(toymodel, sae, input_dim):
    # test how a one-hot vector is encoded
    one_hot_vector = torch.zeros(size=(input_dim,))
    one_hot_vector[0] = 1
    encoding = toymodel.encode(one_hot_vector)
    # random_vector = torch.rand(size=(input_dim,))
    # encoding = toymodel.encode(random_vector)
    print("--- input ---")
    print(encoding)
    print("--- encoding before relu ---")
    print(sae.W_enc @ encoding.squeeze() + sae.bias_enc)
    sae_encoding = sae.encode(encoding)
    print("--- encoding after relu ---")
    print(sae_encoding)
    print("--- decoding ---")
    print(sae.decode(sae_encoding))


def test_reconstruction(toysaemodel):
    input_dim = toysaemodel.toymodel.input_dim
    n_samples = 10000
    inputs = random_sparse_input(input_dim, feature_sparsity=0.8, batch_size=n_samples)
    # print(inputs.shape)
    # encoding = toysaemodel.toymodel.encode(inputs)
    # print(encoding.shape)
    # output = toysaemodel.toymodel.decode(encoding)
    # print(output.shape)


    outputs = toysaemodel(inputs)
    print(outputs.shape)
    outputs = outputs.detach().numpy()

    se = (inputs - outputs)**2
    mse =  torch.mean(se)
    stdev = torch.std(se) / np.sqrt(n_samples) 
    return mse, stdev
