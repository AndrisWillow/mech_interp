# %%
from crosscoder import CrossCoder
import torch
from utils import load_HF_tokenized_DS
from transformer_lens import HookedTransformer
from buffer import Buffer
from constants import BASE_MODEL_NAME, COMPARABLE_MODEL_NAME, default_cfg

torch.set_grad_enabled(False); # for memory reduction
# %%
# Getting scaling factors 

cross_coder = CrossCoder.load_from_hf()
all_tokens = load_HF_tokenized_DS()

device = 'cuda:0'
base_model = HookedTransformer.from_pretrained(
    BASE_MODEL_NAME, 
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    COMPARABLE_MODEL_NAME, 
    device=device, 
)
buffer_instance = Buffer(cfg=default_cfg, model_A=base_model, model_B=chat_model, all_tokens=all_tokens)

base_scaling_factor = buffer_instance.normalisation_factor[0].item()
chat_scaling_factor = buffer_instance.normalisation_factor[1].item()

print("Estimated base scaling factor:", base_scaling_factor)
print("Estimated chat scaling factor:", chat_scaling_factor)
# %%
# Next, we "fold" the scaling factors into the crosscoder's weights.
# This step adjusts the encoding and decoding matrices (and biases) of the crosscoder such that
# they correctly account for the scaling applied to the activations.
def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    # Multiply the encoder weights: the base model's portion gets scaled by its estimated factor,
    # and similarly for the chat model.
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1, :, :] * chat_scaling_factor

    # In the decoder, we perform the inverse: we divide by the scaling factor.
    cross_coder.W_dec.data[:, 0, :] = cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    cross_coder.W_dec.data[:, 1, :] = cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor

    # Similarly adjust the decoder biases.
    cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder
# %%
import copy
# Make a deep copy of the original crosscoder to "fold in" the scaling factors.
folded_cross_coder = copy.deepcopy(cross_coder)
folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_scaling_factor, chat_scaling_factor)
folded_cross_coder = folded_cross_coder.to(torch.bfloat16)
# %%
# Now, compute the cross entropy recovered metrics.
# Here we use the provided function get_ce_recovered_metrics.
# This function:
#   - Computes the clean loss on the base and chat models.
#   - Computes the zero-ablation loss (when the hook activations are zeroed out).
#   - Extracts the residual activations, reconstructs them using the crosscoder,
#     and then "splices" the reconstructed activations back into the forward pass.
#   - Computes the percentage of the cross entropy gap that the reconstruction recovers.
#
# The tokens are selected randomly for demonstration (in practice, you may wish to average over multiple batches).
tokens = all_tokens[torch.randperm(len(all_tokens))[:1]]

from functools import partial
import einops

def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act # Drop BOS # We don't need to drop BOS # TODO adjust whole script to take that into account
    return act

def zero_ablation_hook(act, hook):
    act[:] = 0
    return act

def get_ce_recovered_metrics(tokens, model_A, model_B, cross_coder):
    # get clean loss
    ce_clean_A = model_A(tokens, return_type="loss")
    ce_clean_B = model_B(tokens, return_type="loss")

    # get zero abl loss
    ce_zero_abl_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )
    ce_zero_abl_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )

    # bunch of annoying set up for splicing
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

    cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=0)
    cross_coder_input = cross_coder_input[:, :, 1:, :] # Drop BOS
    cross_coder_input = einops.rearrange(
        cross_coder_input,
        "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
    )

    cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
    cross_coder_output = einops.rearrange(
        cross_coder_output,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model", batch = tokens.shape[0]
    )
    cross_coder_output_A = cross_coder_output[0]
    cross_coder_output_B = cross_coder_output[1]

    # get spliced loss
    ce_loss_spliced_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_A))],
    )
    ce_loss_spliced_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_B))],
    )

    # compute % CE recovered metric
    ce_recovered_A = 1 - ((ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A))
    ce_recovered_B = 1 - ((ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B))

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

tokens = all_tokens[torch.randperm(len(all_tokens))[:1]]
ce_metrics = get_ce_recovered_metrics(tokens, base_model, chat_model, folded_cross_coder)