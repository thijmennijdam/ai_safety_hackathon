#%% Imports
import sys
sys.path.append("../Automatic-Circuit-Discovery")
sys.path.append("..")

import torch as t
import einops
import plotly.express as px

from transformer_lens import HookedTransformer, ActivationCache
from acdc.greaterthan.utils import get_all_greaterthan_things
from utils.prune_utils import get_3_caches

from huggingface_hub import hf_hub_download
from autoencoders.learned_dict import TiedSAE, UntiedSAE

from typing import Literal, Optional, Tuple


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')
hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")

#%% Get transformer model running

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

# %% inspect hook transformer
# print all hook points in model
# make a custom MLP module, that adds hook points for W_in and W_out
# Inspect hook transformer and print all hook points in model
print("All hook points in the model:")
print(model.hook_points)

#%% Get clean and corrupting datasets and task specific metric
BATCH_SIZE = 1 # set to 1, auto encoder is trained for 1, but can be retrained
things = get_all_greaterthan_things(
    num_examples=BATCH_SIZE, metric_name="greaterthan", device=device, model_name="EleutherAI/pythia-70m-deduped"
)
greaterthan_metric = things.validation_metric
clean_ds = things.validation_data # clean data x_i
corr_ds = things.validation_patch_data # corrupted data x_i'

print("\nClean dataset samples")
for stage_cnt in range(1):
    print(model.tokenizer.decode(clean_ds[stage_cnt]))

print("\nReference dataset samples")
for stage_cnt in range(1):
    print(model.tokenizer.decode(corr_ds[stage_cnt]))

#%% Run the model on a dataset sample to verify the setup worked
sample_idx = 0
next_token_logits = model(clean_ds[sample_idx])[-1, -1]
next_token_str = model.tokenizer.decode(next_token_logits.argmax())
print(f"prompt: {model.tokenizer.decode(clean_ds[sample_idx])}")
print(f"next token: {next_token_str}")


# %% Define Hook filters for upstream and downstream nodes
# Upstream nodes in {Embeddings ("blocks.0.hook_resid_pre"), Attn_heads ("result"), MLPs ("mlp_out")}
# Downstream nodes in {Attn_heads ("input") , MLPs ("mlp_in"), resid_final ("blocks.11.hook_resid_post")}
# Necessary Transformerlens flags: model.set_use_hook_mlp_in(True), model.set_use_split_qkv_input(True), model.set_use_attn_result(True)
# upstream_hook_names = ("blocks.0.hook_resid_pre", "hook_result", "hook_mlp_out")
# downstream_hook_names = ("hook_q_input","hook_k_input", "hook_v_input", "hook_mlp_in", "blocks.11.hook_resid_post")

# %% Get the required caches for calculating EAP scores
# (2 forward passes on clean and corr ds, backward pass on clean ds)

def get_3_caches(model,
    clean_input,
    corrupted_input,
    metric,
    mode: Literal["node", "edge"] = "node",
    upstream_hook_names: Optional[Tuple] = None,
    downstream_hook_names: Optional[Tuple] = None,
    ):

    # default hook names
    if not upstream_hook_names:
        upstream_hook_names = ("hook_result", "hook_mlp_out", "blocks.0.hook_resid_pre", "hook_q", "hook_k", "hook_v")
    if not downstream_hook_names:
        if model.cfg.attn_only:   
            downstream_hook_names = ("hook_q_input", "hook_k_input", "hook_v_input", f"blocks.{model.cfg.n_layers-1}.hook_resid_post")
        else:
            downstream_hook_names = ("hook_mlp_in", "hook_q_input", "hook_k_input", "hook_v_input", f"blocks.{model.cfg.n_layers-1}.hook_resid_post")
            
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()

    edge_acdcpp_outgoing_filter = lambda name: name.endswith(upstream_hook_names)
    model.add_hook(hook_filter if mode == "node" else edge_acdcpp_outgoing_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()   

    edge_acdcpp_back_filter = lambda name: name.endswith(downstream_hook_names)
    model.add_hook(hook_filter if mode=="node" else edge_acdcpp_back_filter, backward_cache_hook, "bwd")
    value = metric(model(clean_input))
    value.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_corrupted_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()

    model.add_hook(hook_filter if mode == "node" else edge_acdcpp_outgoing_filter, forward_corrupted_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache


upstream_hook_names = ("blocks.0.hook_resid_pre", "hook_result", "hook_mlp_out")
downstream_hook_names = ("hook_q_input","hook_k_input", "hook_v_input", "hook_mlp_in", "blocks.11.hook_resid_post")

clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(
    model,
    clean_ds,
    corr_ds,
    greaterthan_metric,
    mode="edge",
    upstream_hook_names=upstream_hook_names,
    downstream_hook_names=downstream_hook_names
)

# %% Compute matrix holding all attribution scores
# edge_attribution_score = (upstream_corr - upstream_clean) * downstream_grad_clean
N_UPSTREAM_STAGES = len(clean_cache)
N_DOWNSTREAM_STAGES = len(clean_grad_cache) - 2*model.cfg.n_layers # qkv
SEQUENCE_LENGTH = clean_ds.shape[1]

N_TOTAL_UPSTREAM_NODES = 1 + model.cfg.n_layers * (model.cfg.n_heads + 1)
N_TOTAL_DOWNSTREAM_NODES = 1 + model.cfg.n_layers * (3*model.cfg.n_heads + 1)

# Get (upstream_corr - upstream_clean) as matrix
N_UPSTREAM_NODES = model.cfg.n_heads
upstream_cache_clean = t.zeros((
    N_TOTAL_UPSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))
upstream_cache_corr = t.zeros_like(upstream_cache_clean)

upstream_names = []
upstream_levels = t.zeros(N_TOTAL_UPSTREAM_NODES)
idx = 0
for stage_cnt, name in enumerate(clean_cache.keys()): # stage_cnt relevant for keeping track which upstream-downstream mairs can be connected
    if name.endswith("result"): # layer of attn heads
        act_clean = einops.rearrange(clean_cache[name], "b s nh dm -> nh b s dm")
        act_corr = einops.rearrange(corrupted_cache[name], "b s nh dm -> nh b s dm")
        upstream_cache_clean[idx:idx+model.cfg.n_heads] = act_clean
        upstream_cache_corr[idx:idx+model.cfg.n_heads] = act_corr
        upstream_levels[idx:idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = name + str(i)
            upstream_names.append(head_name)
    else:
        upstream_cache_clean[idx] = clean_cache[name]
        upstream_cache_corr[idx] = corrupted_cache[name]
        upstream_levels[idx] = stage_cnt
        idx += 1
        upstream_names.append(name)

upstream_diff = upstream_cache_corr - upstream_cache_clean

#%% Get downstream_grad as matrix
N_DOWNSTREAM_NODES = model.cfg.n_heads * 3 # q, k, v separate
downstream_grad_cache_clean = t.zeros((
    N_TOTAL_DOWNSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))

downstream_names = []
downstream_levels = t.zeros(N_TOTAL_DOWNSTREAM_NODES)
stage_cnt = 0
idx = 0
names = reversed(list(clean_grad_cache.keys()))
for name in names:
    if name.endswith("hook_q_input"): # do all q k v hooks of that layer simultaneously, as it is the same stage
        q_name = name
        k_name = name[:-7] + "k_input"
        v_name = name[:-7] + "v_input"
        q_act = einops.rearrange(clean_grad_cache[q_name], "b s nh dm -> nh b s dm")
        k_act = einops.rearrange(clean_grad_cache[k_name], "b s nh dm -> nh b s dm")
        v_act = einops.rearrange(clean_grad_cache[v_name], "b s nh dm -> nh b s dm")

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = q_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = q_name + str(i)
            downstream_names.append(head_name)
        
        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = k_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = k_name + str(i)
            downstream_names.append(head_name)

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = v_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = v_name + str(i)
            downstream_names.append(head_name)

    elif name.endswith(("hook_k_input", "hook_v_input")):
        continue
    else:
        downstream_grad_cache_clean[idx] = clean_grad_cache[name]
        downstream_levels[idx] = stage_cnt
        idx += 1
        downstream_names.append(name)
    stage_cnt += 1

#%% Calculate the cartesian product of stage, node for upstream and downstream
eap_scores = einops.einsum(
    upstream_diff, 
    downstream_grad_cache_clean,
    "up_nodes batch seq d_model, down_nodes batch seq d_model -> up_nodes down_nodes"
)



#%% Make explicit only upstream -> downstream (not downstream -> upstream is important)
upstream_level_matrix = einops.repeat(upstream_levels, "up_nodes -> up_nodes down_nodes", down_nodes=N_TOTAL_DOWNSTREAM_NODES)
downstream_level_matrix = einops.repeat(downstream_levels, "down_nodes -> up_nodes down_nodes", up_nodes=N_TOTAL_UPSTREAM_NODES)
mask = upstream_level_matrix > downstream_level_matrix
eap_scores = eap_scores.masked_fill(mask, value=t.nan)

# add final node to downstream as it is not in the grad cache
downstream_names.append("resid_final")

print(len(downstream_names), eap_scores.shape)

# px.imshow(
#     eap_scores,
#     x=downstream_names,
#     y=upstream_names,
#     labels = dict(x="downstream node", y="upstream node", color="EAP score"),
#     color_continuous_scale="RdBu",
#     color_continuous_midpoint=0
# )

# %% Load in SAEs for model
from huggingface_hub import hf_hub_download
from autoencoders.learned_dict import TiedSAE, UntiedSAE


model_id = "jbrinkma/Pythia-70M-deduped-SAEs"
ae_download_location = hf_hub_download(repo_id=model_id, filename=f"Pythia-70M-deduped-mlp-0.pt")
ae = t.load(ae_download_location, map_location=t.device('cpu'))

mlp_count = model.cfg.n_layers
print("Loading SAEs for", mlp_count, "MLPs")
sae_list = []
for i in range(mlp_count):
    ae_download_location = hf_hub_download(repo_id=model_id, filename=f"Pythia-70M-deduped-mlp-{i}.pt")
    sae = t.load(ae_download_location, map_location=t.device('cpu'))
    sae_list.append(sae)
print("Done")

# %% Hook mlp_post for all layers and use sae to encode/decode
model.reset_hooks()

def hook_mlp_post(act, hook):
    layer_num = int(hook.name.split(".")[1])
    sae = sae_list[layer_num]

    # encode and decode using sae
    print("act", act.shape)
    act = act.squeeze(0) # squeeze to drop batch dim, as pretrained SAEs expect no batch dim
    print("act (squeezed)", act.shape)
    act = sae.encode(act)
    print("act (encoded)", act.shape)
    act = sae.decode(act)
    print("act (decoded)", act.shape)
    act = act.unsqueeze(0) # add back batch dim
    print("act (unsqueezed)", act.shape)
    return act

# Add hooks to all mlps in the model
for i in range(mlp_count):
    model.add_hook(f"blocks.{i}.mlp.hook_pre", hook_mlp_post, "fwd")

# %% Run the model on dataset samples to verify the SAE recreation works
for i in range(len(clean_ds)):
    next_token_logits = model(clean_ds[i])[-1, -1]
    next_token_str = model.tokenizer.decode(next_token_logits.argmax())
    print(f"prompt: {model.tokenizer.decode(clean_ds[i])}")
    print(f"next token: {next_token_str}")  

# %% cache mlp intermediate gradients for downstream nodes
downstream_mlp_grad_cache = t.zeros((
    N_TOTAL_DOWNSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))