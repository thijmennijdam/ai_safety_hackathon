#%% Imports
import sys
sys.path.append("../Automatic-Circuit-Discovery")
sys.path.append("..")

import torch as t
import einops
import plotly.express as px

from transformer_lens import HookedTransformer
from acdc.greaterthan.utils import get_all_greaterthan_things
from utils.prune_utils import get_3_caches

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')

#%% Get pythia-70m transformer model running
# Get pythia-70m transformer model running
model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-70m-deduped',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

#%% inspect hooks
print("All hook points in the model:")
print(model.hook_points)
# %% create input for model
input_ids = model.tokenizer.encode("Hello I am 10 years old, thats 5 + ", return_tensors="pt").to(device)

# %% run input through model and print output
# run input through model, detokenize and print
tokens = model.generate(input_ids)
print(model.tokenizer.decode(tokens[0]))


#%% inspect model and print layers shapes
print("Model layers:")
print(model.state_dict().keys())
print("Model layers shapes:")
for k, v in model.state_dict().items():
    print(k, v.shape)
# %%
import sys
sys.path.append("..")
import torch as t
from huggingface_hub import hf_hub_download
from autoencoders.learned_dict import TiedSAE, UntiedSAE

layer = 0
model_id = "jbrinkma/Pythia-70M-deduped-SAEs"
# model_id = "Elriggs/pythia-70M-deduped-sae"
# model_id = "Elriggs/pythia-70m-deduped"
ae_download_location = hf_hub_download(repo_id=model_id, filename=f"Pythia-70M-deduped-mlp-0.pt")
# ae_download_location = hf_hub_download(repo_id=model_id, filename=f"pythia-70m-deduped_r4_gpt_neox.layers.0.pt")
# ae_download_location = hf_hub_download(repo_id=model_id, filename=f"tied_residual_l0_r6/_0/learned_dicts.pt")
ae = t.load(ae_download_location, map_location=t.device('cpu'))

# %% inspect hooked transfomer
model.reset_hooks()

# %% inspect model layers
# print(model.state_dict()["blocks.0.mlp.W_in"].shape)

def hook_pre(act, hook):
    # run through ae encode
    print("act shape", act.shape)
    act = act.squeeze(0)
    print("act shape (squeeze)", act.shape)
    output = ae.encode(act)
    decoded_input = ae.decode(output)
    print("decoded", decoded_input.shape)
    print("decoded (unsqueezed)", decoded_input.unsqueeze(0).shape)
    return decoded_input.unsqueeze(0)

model.add_hook("blocks.0.hook_mlp_out", hook_pre, "fwd")

# # hook the mlp hook_post hookpoint
# def hook_post(act, hook):
#     decoded_input = ae.decode(act)
#     return decoded_input
# model.add_hook("blocks.0.hook_mlp_in", hook_post, "fwd")

# %% add a pre_hook for the model.blocks[0].mlp, and print the activations

tokens = model.generate(input_ids)
print(model.tokenizer.decode(tokens[0]))




# %%
