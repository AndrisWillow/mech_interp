import torch
import os
from functools import partial
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from transformers import AutoTokenizer, BitsAndBytesConfig

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"


from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=device,
)

from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner

test_feature_idx_gpt = list(range(10)) + [14057]
hook_name = sae.cfg.hook_name
feature_vis_config_gpt = SaeVisConfig(
    hook_point=hook_name,
    features=test_feature_idx_gpt,
    minibatch_size_features=64,
    minibatch_size_tokens=256,
    verbose=True,
    device=device,
)

visualization_data_gpt = SaeVisRunner(
    feature_vis_config_gpt
).run(
    encoder=sae,  # type: ignore
    model=model,
    tokens=token_dataset[:10000]["tokens"],  # type: ignore
)
# SaeVisData.create(
#     encoder=sae,
#     model=model, # type: ignore
#     tokens=token_dataset[:10000]["tokens"],  # type: ignore
#     cfg=feature_vis_config_gpt,
# )

from sae_dashboard.data_writing_fns import save_feature_centric_vis

filename = f"demo_feature_dashboards.html"
save_feature_centric_vis(sae_vis_data=visualization_data_gpt, filename=filename)