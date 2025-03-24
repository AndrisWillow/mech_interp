# Source: https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html#Setup

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px

from jaxtyping import Float
from functools import partial
import circuitsvis as cv

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

torch.set_grad_enabled(False) # Saves some memory since we are using model inference not training

device = utils.get_device()
print(f"Device: {device}")

# meta-llama/Llama-3.2-1B-Instruct # meta-llama/Llama-3.2-1B
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device=device)

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)

# Visulization test:

model_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
model_tokens = model.to_tokens(model_text)
print(model_tokens.device)
model_logits, model_cache = model.run_with_cache(model_tokens, remove_batch_dim=True)

print(type(model_cache))
attention_pattern = model_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
model_str_tokens = model.to_str_tokens(model_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(tokens=model_str_tokens, attention=attention_pattern)

attn_hook_name = "blocks.0.attn.hook_pattern"
attn_layer = 0
_, llama_attn_cache = model.run_with_cache(model_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
llama_attn = llama_attn_cache[attn_hook_name]
assert torch.equal(llama_attn, attention_pattern)


# Ablation test:

# layer_to_ablate = 0
# head_index_to_ablate = 8

# # We define a head ablation hook
# # The type annotations are NOT necessary, they're just a useful guide to the reader


# def head_ablation_hook(
#     value: Float[torch.Tensor, "batch pos head_index d_head"],
#     hook: HookPoint
# ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
#     print(f"Shape of the value tensor: {value.shape}")
#     value[:, :, head_index_to_ablate, :] = 0.
#     return value

# original_loss = model(model_tokens, return_type="loss")
# ablated_loss = model.run_with_hooks(
#     model_tokens,
#     return_type="loss",
#     fwd_hooks=[(
#         utils.get_act_name("v", layer_to_ablate),
#         head_ablation_hook
#         )]
#     )
# print(f"Original Loss: {original_loss.item():.3f}")
# print(f"Ablated Loss: {ablated_loss.item():.3f}")