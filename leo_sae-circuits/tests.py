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


def test_reconstruction(toysaemodel, one_feature=False):
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

    if one_feature:
        # define se for one feature
        se = (inputs[:, 0] - outputs[:, 0])**2
    else:
        se = (inputs - outputs)**2
    
    mse =  torch.mean(se)
    mse.backward()
    
    print("gradients of feature importance")
    stdev = torch.std(se) / np.sqrt(n_samples) 
    return mse, stdev

def test_reconstruction_one_feature(toysaemodel):
    input_dim = toysaemodel.toymodel.input_dim
    # n_samples = 50
    one_hot_vector = torch.zeros(size=(input_dim,))
    one_hot_vector[0] = 1
    inputs = one_hot_vector    
    
    # inputs = random_sparse_input(input_dim, feature_sparsity=0.8, batch_size=one_hot_vector)
    
    # use retain_grad()
    inputs.requires_grad = True
    
    # forward pass through ToyModel and SAE
    features = toysaemodel.sae.encode(toysaemodel.toymodel.encode(inputs))
    features.retain_grad()
    # decode all the way back to the input
    outputs = toysaemodel.toymodel.decode(toysaemodel.sae.decode(features))

    
    
    loss = torch.mean((features[:, 0] - outputs[:, 0])**2)
    
    loss.backward()
    
    print("loss:", loss)
    print("grad of loss with respect to the feature")
    print(features.grad)
    
    print(features.grad.shape)
    # print("features before backward pass")
    # # print(features.grad)
    # # backward pass

    # # accessing the gradients with respect to the input
    # gradients = inputs.grad
    # grad_feature = features.grad
    
    # print("Gradients with respect to the input:", gradients)
    
    return features, loss
