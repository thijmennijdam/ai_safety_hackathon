import torch
import inspect
import shutil
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer
import tqdm

model = HookedTransformer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

input_str = "I just want to let you know that cats are"

inputs = tokenizer(input_str, return_tensors="pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

type(model)

# model.to(device)
# inputs.to(device)

# max_length = 100
# assert max_length > len(input_str)

# tokens = model.generate(**inputs, max_length=max_length)
# output_str = tokenizer.decode(tokens[0])

# print(input_str)
# print(output_str)

prompt1 = "I just want to let you know that cats are"
prompt1_tokens = model.to_tokens(prompt1)


# We define a residual stream patching hook
# We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# The type annotations are a guide to the reader and are not necessary
def residual_stream_patching_hook(
    resid_pre,
    hook: HookPoint,
):
    # Each HookPoint has a name attribute giving the name of the hook.
    # clean_resid_pre = clean_cache[hook.name]
    # resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

# We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
num_positions = len(prompt1_tokens[0])

for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    # Run the model with the patching hook
    logits = model.run_with_hooks(prompt1_tokens, fwd_hooks=[
        (utils.get_act_name("resid_pre", layer), residual_stream_patching_hook)
    ])

