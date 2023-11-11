import torch
import matplotlib.pyplot as plt

def random_sparse_input(input_dim, feature_sparsity, batch_size=1, device="cpu"):
    # Create a random tensor with an additional batch dimension
    x = torch.rand(size=(batch_size, input_dim), device=device)
    # Randomly zero out some features with the probability of feature_sparsity
    sparse_x = torch.where(torch.rand_like(x) < feature_sparsity, torch.zeros_like(x), x)
    return sparse_x

def test_random_sparse_input():
    input_dim = 20
    feature_sparsity = 0.8
    for i in range(10):
        x = random_sparse_input(input_dim, feature_sparsity)
        print(x)

def plot_inputs_and_outputs(input_dim, toymodel):
    # Create a figure with 2 rows (for input and output) and 'num_items' columns
    # input_dim = 10  # Replace with your actual input dimension

    def plot_one_hot_vector(index):
        # Create a subplot grid with 10 rows and 2 columns
        plt.subplot(10, 2, 2*index + 1)
        
        # One-hot vector
        x = torch.zeros(size=(input_dim,))
        x[index] = 1
        
        plt.imshow(x.numpy().reshape(1, -1))
        plt.title(f"Input {index}")
        
        plt.subplot(10, 2, 2*index + 2)
        
        # Compute the output using your toymodel
        output = toymodel.forward(x)
        
        plt.imshow(output.cpu().detach().numpy().reshape(1, -1))
        plt.title(f"Output {index}")

    # Create a figure for the plot
    plt.figure(figsize=(10, 20))

    for i in range(input_dim):
        plot_one_hot_vector(i)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_inputs_and_sae_encoding(sae, toy_model):
    # Create a figure with 2 rows (for input and output) and 'num_items' columns
    # input_dim = 10  # Replace with your actual input dimension

    def plot_one_hot_vector(index):
        # Create a subplot grid with 10 rows and 2 columns
        plt.subplot(10, 2, 2*index + 1)
        
        # One-hot vector
        x = torch.zeros(size=(toy_model.input_dim,))
        x[index] = 1
        
        plt.imshow(x.numpy().reshape(1, -1))
        plt.title(f"Input {index}")
        
        plt.subplot(10, 2, 2*index + 2)
        
        # Compute the output using your toymodel
        y = toy_model.encode(x)
        print(y.shape)
        h_sae = sae.encode(y)
        
        plt.imshow(h_sae.cpu().detach().numpy().reshape(1, -1))
        plt.title(f"Hidden{index}")

    # Create a figure for the plot
    plt.figure(figsize=(10, 20))

    for i in range(toy_model.input_dim):
        plot_one_hot_vector(i)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
