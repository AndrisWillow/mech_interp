# %%
from crosscoder import CrossCoder
import torch
from utils import load_HF_tokenized_DS
from transformer_lens import HookedTransformer
from buffer import Buffer
from constants import default_cfg
import copy
from functools import partial
import einops
import tqdm
import numpy as np

torch.set_grad_enabled(False)  # Disable gradients to reduce memory usage.
# %%
# Load the crosscoder and both the base and comparable (chat) models.
cross_coder = CrossCoder.load_from_hf()

# Load tokenized dataset (all_tokens could be very large, but our Buffer uses a subset for scaling).
all_tokens = load_HF_tokenized_DS(as_tensor=True) # TODO (out of RAM issues!) Modify the code so it doesn't have to rely on loading all of the tokens at once

device = 'cuda:0'
base_model = HookedTransformer.from_pretrained(
    default_cfg["base_model"], 
    device=device,
    dtype=torch.bfloat16, 
)
chat_model = HookedTransformer.from_pretrained(
    default_cfg["comperable_model"], 
    device=device,
    dtype=torch.bfloat16, 
)

# %%
# Create a Buffer instance. 
# The Buffer will use a subsample (default 100 batches) from all_tokens to estimate the norm scaling
# factors for each model. Even if all_tokens is huge, only a small, representative subset is used.
buffer_instance = Buffer(cfg=default_cfg, model_A=base_model, model_B=chat_model, all_tokens=all_tokens, refresh_buffer=False)
# TODO disable buffer refreshing
# Retrieve the estimated scaling factors for each model.
base_scaling_factor = buffer_instance.normalisation_factor[0].item()
chat_scaling_factor = buffer_instance.normalisation_factor[1].item()

print("Estimated base scaling factor:", base_scaling_factor)
print("Estimated chat scaling factor:", chat_scaling_factor)

# %%
# Next, "fold" the scaling factors into the crosscoder's weights.
# This adjusts the encoder and decoder weights (and biases) to ensure that the activations 
# are normalized to have an average norm of sqrt(d_model) when processed by the crosscoder.
def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    # Scale the encoder weights appropriately for each model slice.
    cross_coder.W_enc.data[0, :, :] *= base_scaling_factor
    cross_coder.W_enc.data[1, :, :] *= chat_scaling_factor

    # For the decoder, reverse the scaling by dividing.
    cross_coder.W_dec.data[:, 0, :] /= base_scaling_factor
    cross_coder.W_dec.data[:, 1, :] /= chat_scaling_factor

    # Adjust decoder biases accordingly.
    cross_coder.b_dec.data[0, :] /= base_scaling_factor
    cross_coder.b_dec.data[1, :] /= chat_scaling_factor
    return cross_coder

# Make a deep copy of the crosscoder to apply the scaling factors.
folded_cross_coder = copy.deepcopy(cross_coder)
folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_scaling_factor, chat_scaling_factor)
folded_cross_coder = folded_cross_coder.to(torch.bfloat16)

# %%
# Define hook functions for splicing and zero ablation.
# Since we are no longer dropping the BOS token, we simply replace the activations with the spliced ones.
def splice_act_hook(act, hook, spliced_act):
    # Replace the entire activation with the spliced (reconstructed) one.
    return spliced_act

def zero_ablation_hook(act, hook):
    # Set all activations to zero.
    return torch.zeros_like(act)

# %%
# Define the function to compute cross entropy (CE) recovered metrics.
# This function:
# - Computes the clean CE loss for both models.
# - Computes the zero-ablation loss (i.e. CE when the activation at the hook is zeroed out).
# - Extracts the residual activations from both models.
# - Uses the crosscoder to reconstruct the activations.
# - "Splices" the reconstructed activations back into the forward pass.
# - Computes the percentage of the CE gap that is recovered by the spliced reconstruction.
def get_ce_recovered_metrics(tokens, model_A, model_B, cross_coder):
    # Compute clean losses.
    ce_clean_A = model_A(tokens, return_type="loss")
    ce_clean_B = model_B(tokens, return_type="loss")

    # Compute losses when the hook activations are zeroed (zero-ablation).
    ce_zero_abl_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )
    ce_zero_abl_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )

    # Extract residual activations from both models at the desired hook.
    _, cache_A = model_A.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_A = cache_A[cross_coder.cfg["hook_point"]]

    _, cache_B = model_B.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_B = cache_B[cross_coder.cfg["hook_point"]]

    # Stack the activations for both models.
    cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=0)
    # Do not drop the BOS token – use the full sequence:
    # cross_coder_input = cross_coder_input[:, :, 1:, :]  --> Removed
    
    # Rearrange: Combine batch and sequence dimensions for processing.
    cross_coder_input = einops.rearrange(
        cross_coder_input,
        "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
    )

    # Run through the crosscoder.
    cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
    # Rearrange output back to separate model, batch, and sequence dimensions.
    cross_coder_output = einops.rearrange(
        cross_coder_output,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model", 
        batch=tokens.shape[0]
    )
    cross_coder_output_A = cross_coder_output[0]
    cross_coder_output_B = cross_coder_output[1]

    # Compute the loss when activations are spliced with crosscoder outputs.
    ce_loss_spliced_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_A))],
    )
    ce_loss_spliced_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_B))],
    )

    # Compute percentage of CE gap recovered by the crosscoder.
    ce_recovered_A = 1 - ((ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A))
    ce_recovered_B = 1 - ((ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B))

    # Return all relevant metric values in a dictionary.
    metrics = {
        "ce_loss_spliced_A": ce_loss_spliced_A.item(),
        "ce_loss_spliced_B": ce_loss_spliced_B.item(),
        "ce_clean_A": ce_clean_A.item(),
        "ce_clean_B": ce_clean_B.item(),
        "ce_zero_abl_A": ce_zero_abl_A.item(),
        "ce_zero_abl_B": ce_zero_abl_B.item(),
        "ce_diff_A": (ce_loss_spliced_A - ce_clean_A).item(),
        "ce_diff_B": (ce_loss_spliced_B - ce_clean_B).item(),
        "ce_recovered_A": ce_recovered_A.item(),
        "ce_recovered_B": ce_recovered_B.item(),
    }
    return metrics
# %%
# Instead of using one batch from all_tokens, we average the metrics over multiple randomly chosen examples.
num_examples = 5  # Adjust this number as needed for a more stable estimate.
indices = torch.randperm(len(all_tokens))[:num_examples]
all_metrics = []

for idx in indices:
    # Assume that each selected index corresponds to a valid token batch.
    tokens = all_tokens[idx:idx + 1]  # Maintain the batch dimension.
    metrics = get_ce_recovered_metrics(tokens, base_model, chat_model, folded_cross_coder)
    all_metrics.append(metrics)

# Average each metric over the examples.
avg_metrics = {}
for key in all_metrics[0].keys():
    avg_metrics[key] = np.mean([m[key] for m in all_metrics])

print("Averaged CE recovered metrics:")
for k, v in avg_metrics.items():
    print(f"{k}: {v:.4f}")

# %%
!pip install git+https://github.com/ckkissane/sae_vis.git@crosscoder-vis
# %%
# SAE VIS TODO

import copy
folded_cross_coder = copy.deepcopy(cross_coder)

def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] *= base_scaling_factor
    cross_coder.W_enc.data[1, :, :] *= chat_scaling_factor
    
    return cross_coder

folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_scaling_factor, chat_scaling_factor)
# %%
from sae_vis.model_fns import CrossCoderConfig, CrossCoder

encoder_cfg = CrossCoderConfig(d_in=base_model.cfg.d_model, d_hidden=cross_coder.cfg["dict_size"], apply_b_dec_to_input=False)
sae_vis_cross_coder = CrossCoder(encoder_cfg)
sae_vis_cross_coder.load_state_dict(folded_cross_coder.state_dict())
sae_vis_cross_coder = sae_vis_cross_coder.to("cuda:0")
sae_vis_cross_coder = sae_vis_cross_coder.to(torch.bfloat16)
sae_vis_cross_coder = folded_cross_coder.to("cuda:0").to(torch.bfloat16)
# %%
# Latent ids gotten from 1_analysis_hist
IT_specific_latents=[107, 578, 658, 895, 1559, 1859, 2003, 2217, 2442, 2761, 2851, 3042, 3406, 3416, 3442, 3456, 3595, 3777, 3871, 3896, 3953, 3985, 4259, 4290, 4291, 4568, 4608, 5074, 5144, 5439, 5479, 5480, 5493, 5676, 5797, 5899, 6009, 6098, 6141, 6251, 6524, 6701, 6929, 7069, 7370, 7412, 7417, 7608, 7625, 7659, 7924, 7954, 8065, 8071, 8114, 8178, 8200, 8204, 8321, 8398, 8420, 8658, 8777, 8780, 8800, 8812, 8823, 8830, 8888, 8892, 9008, 9242, 9393, 9631, 9839, 9899, 10035, 10100, 10332, 10458, 10577, 10665, 10729, 10807, 11017, 11167, 11589, 11646, 11703, 11762, 11883, 12002, 12310, 12901, 13117, 13210, 13228, 13323, 13332, 13443, 13580, 13593, 13725, 13798, 14058, 14355, 14401, 14417, 14571, 14573, 14906, 15000, 15100, 15212, 15823, 15931, 16015]
from sae_vis.data_config_classes import SaeVisConfig
test_feature_idx = IT_specific_latents
sae_vis_config = SaeVisConfig(
    hook_point = folded_cross_coder.cfg["hook_point"],
    features = test_feature_idx,
    verbose = True,
    # Max for RTX3090
    minibatch_size_tokens=6,
    minibatch_size_features=24,
)
# %%
from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = sae_vis_cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = chat_model,
    tokens = all_tokens[:1024], # in practice, better to use more data
    cfg = sae_vis_config,
)
# %%
import os
import http
import socketserver
import threading
import webbrowser

PORT = 8000

def display_vis_inline(filename: str, height: int = 850):
    """
    Launches a local HTTP server to serve files from the current working directory,
    then opens the specified file URL (http://localhost:PORT/filename) in the default web browser.
    """
    global PORT

    # This inner function serves files from a specified directory.
    def serve(directory):
        os.chdir(directory)
        handler = http.server.SimpleHTTPRequestHandler
        # Create and run a TCP server that listens on PORT.
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"Serving files from {directory} on port {PORT}")
            httpd.serve_forever()

    # Start the HTTP server in a separate daemon thread.
    thread = threading.Thread(target=serve, args=(os.getcwd(),))
    thread.setDaemon(True)
    thread.start()

    # Build the URL for the file and open it in the default web browser.
    url = f"http://localhost:{PORT}/{filename}"
    print(f"Opening URL: {url}")
    webbrowser.open(url)

    PORT += 1

# %%
# Save the visualization HTML file.
filename = "_feature_vis_demo_llama-1024-itSpecific.html"
sae_vis_data.save_feature_centric_vis(filename)

# Display the visualization by opening it in a browser.
display_vis_inline(filename)
# %%
