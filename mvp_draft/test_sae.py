from huggingface_hub import hf_hub_download
import torch

layer = 2
model_id = "Elriggs/pythia-70m-deduped"
# model_id = "Elriggs/pythia-410m-deduped"
ae_download_location = hf_hub_download(repo_id=model_id, filename=f"tied_residual_l{layer}_r6/_63/learned_dicts.pt")
all_autoencoders = torch.load(ae_download_location)
num_l1s = len(all_autoencoders)
all_l1s = [hyperparams["l1_alpha"] for autoencoder, hyperparams in all_autoencoders]
print(all_l1s)
auto_num = 5
autoencoder, hyperparams = all_autoencoders[auto_num]
# You want a hyperparam around 1e-3. Higher is less features/datapoint (at the cost of reconstruction error); lower is more features/datapoint (at the cost of polysemanticity)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to_device(device)
print(hyperparams)

ae = all_autoencoders[0][0]

help(ae)

dir(ae)
ae.activation_size
ae.decode
ae.encode
ae.get_learned_dict
ae.n_dict_components
ae.n_feats
