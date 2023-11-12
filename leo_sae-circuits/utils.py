import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def random_sparse_input(input_dim, feature_sparsity, batch_size=1, device="cpu"):
    # Create a random tensor with an additional batch dimension
    x = torch.rand(size=(batch_size, input_dim), device=device)
    # Randomly zero out some features with the probability of feature_sparsity
    sparse_x = torch.where(torch.rand_like(x) < feature_sparsity, torch.zeros_like(x), x)
    return sparse_x

# def random_sparse_input_with_one_hot_feature(input_dim, batch_size=1, device="cpu"):
#     # one hot vector
#     x = torch.zeros(size=(batch_size, input_dim), device=device)
#     # random index
#     index = torch.randint(low=0, high=input_dim, size=(batch_size,))
#     # set the random index to 1
#     x[torch.arange(batch_size), index] = 1
#     # random value
#     value = torch.rand(size=(batch_size,))
#     # set the random value to the random index


def test_random_sparse_input():
    input_dim = 20
    feature_sparsity = 0.8
    for i in range(10):
        x = random_sparse_input(input_dim, feature_sparsity)
        print(x)

def one_hot_vector(index, input_dim):
    x = torch.zeros(size=(input_dim,))
    x[index] = 1
    return x

def plot_inputs_and_outputs(input_dim, toymodel):
    """
    Plot the input and output of the toymodel for all possible inputs.
    """
    model_function = toymodel.forward
    model_function_name = "Toy Model output"
    plot_model_function_for_all_input_dims(input_dim, model_function, model_function_name)

def plot_inputs_and_sae_encoding(sae, toy_model):
    """
    Plot the SAE encoding for all possible inputs.
    """
    model_function = lambda x: sae.encode(toy_model.encode(x))
    model_function_name = "SAE features"
    plot_model_function_for_all_input_dims(toy_model.input_dim, model_function, model_function_name)
    
    # find_reordering_of_features(model_function, toy_model.input_dim)

def plot_model_function_for_all_input_dims(input_dim, model_function, model_function_name):
    _, axes = plt.subplots(input_dim, 2,)  # Increase figure size

    reordered_index = find_reordering_of_features(model_function, input_dim)
    print(reordered_index)
    
    for i in range(input_dim):
        ax1 = axes[i, 0]
        x = one_hot_vector(i, input_dim)
        ax1.imshow(x.numpy().reshape(1, -1), cmap='viridis', aspect='auto')  # Adjust colormap and aspect ratio
        ax1.axis('off')

        ax2 = axes[i, 1]
        # print("---- input ----")
        # print(x.shape)
        reordered_x = x[reordered_index]
        # print(reordered_x.shape)
        ax2.imshow(model_function(reordered_x).detach().numpy().reshape(1, -1), cmap='viridis', aspect='auto')  # Adjust colormap and aspect ratio
        ax2.axis('off')
        # print('---- previous plot ----')
        # print(model_function(x).detach().numpy().reshape(1, -1))

    plt.subplots_adjust(wspace=0.05, hspace=0.1)  # Adjust the spacing between subplots
    plt.suptitle(f"Input and {model_function_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


def get_matrix_of_features(model_function, input_dim):
    """
    Get the matrix of features for all possible inputs.
    """
    identity = torch.eye(input_dim)
    sae_features = model_function(identity)
    identity = identity.detach().numpy()
    sae_features = sae_features.detach().numpy()
    return identity, sae_features


def find_reordering_of_features(model_function, input_dim):
    """
    Find the reordering of features that the SAE has learned.
    """
    identity, sae_features = get_matrix_of_features(model_function, input_dim)
    return find_permutation_to_align_features_with_input(identity, sae_features)


def find_permutation_to_align_features_with_input(first_matrix, second_matrix):
    # Compute the cosine similarity between columns of the two matrices
    # We want to minimize distance, which is 1 - similarity
    # print("---- first_matrix ----")
    # print(first_matrix)
    # print("---- second_matrix ----")
    # print(second_matrix)
    distance_matrix = cdist(first_matrix.T, second_matrix.T, 'cosine')
    
    # Solve the linear assignment problem (Hungarian algorithm)
    # This finds the minimum cost assignment between columns of first_matrix and second_matrix
    # print("---- distance_matrix ----")
    # print(distance_matrix)

    _, col_permutation = linear_sum_assignment(distance_matrix)
    # print("---- row_ind ----")
    # print(row_ind)
    # print("---- col_permutation ----")
    # print(col_permutation)
    return col_permutation
    
    # Reorder the second matrix according to the found permutation
    # reordered_second_matrix = second_matrix[:, col_permutation]
    
    # return reordered_second_matrix

# def find_reordering_of_features(first_matrix, second_matrix):
#     distances = cdist(first_matrix.T, second_matrix.T, 'cosine')
    
#     # Find the permutation of columns in the second matrix that minimizes the distance
#     col_permutation = np.argmin(distances, axis=1)
    
#     # Reorder the second matrix according to the found permutation
#     reordered_second_matrix = second_matrix[:, col_permutation]
    
#     return reordered_second_matrix

    # Find the indices of the features sorted by their magnitude
    # _, sorted_indices = torch.sort(torch.abs(sae_features), descending=True)
    # return sorted_indices

# To measure how well the learned dictionary recovered the ground truth features, we took the ‘mean max cosine similarity’ (MMCS) between the ground truth features and the learned dictionary. 

# Intuitively, to calculate the MMCS, we first calculate the cosine similarity between all pairs of dictionary elements in the learned and ground truth dictionaries.

# def metric_mean_max_cosine_similarity()
# https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition


def visualize_feature_gradients(feature_gradients, n_vectors=10):
    """
    Visualize the first `n_vectors` feature gradient vectors in a grid.
    :param feature_gradients: A PyTorch tensor of gradients of shape (n_samples, feature_dim).
    :param n_vectors: Number of gradient vectors to visualize.
    """
    # Convert the gradients to numpy for visualization
    gradients_np = feature_gradients.detach().cpu().numpy()

    # Select the first `n_vectors` for visualization
    selected_gradients = gradients_np[:n_vectors]

    # Number of subplots needed
    n_cols = min(n_vectors, 5)  # Limit columns to 5 for readability
    n_rows = int(np.ceil(n_vectors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_vectors > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n_vectors:
            ax.imshow(np.atleast_2d(selected_gradients[i]), aspect='auto', cmap='viridis')
            ax.set_title(f'Gradient Vector {i+1}')
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Gradient Value')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    