from transformer_lens import ActivationCache
import tqdm
import torch
import einops
import numpy as np

class Buffer:
    """
    This class defines a data buffer that stores a stack of activations (acts)
    from two models (model_A and model_B). These activations can be used to
    train an autoencoder, crosscoder, or for mechanistic interpretability.
    
    Key functionalities:
    - Preallocate a buffer based on configuration parameters.
    - Estimate norm scaling factors for each model using a small subset of tokens.
      These scaling factors are computed such that the average activation norm is ~√(d_model).
    - Optionally auto-populate (refresh) the buffer.
    
    The constructor accepts a new flag `refresh_buffer` (default True). If set to False,
    the scaling factors are computed but the buffer will not be refreshed (populated)
    automatically.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens, refresh_buffer: bool = True):
        # Ensure both models have the same hidden dimension.
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        
        # Compute the total buffer size as a multiple of batch_size and sequence length.
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"])
        self.buffer_size = self.buffer_batches * (cfg["seq_len"])
        
        # Preallocate the activation buffer.
        # The shape is [buffer_size, 2, d_model], where "2" is for model_A and model_B.
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])
        
        # Save model and data references.
        self.model_A = model_A
        self.model_B = model_B
        self.all_tokens = all_tokens
        
        # Initialize pointers and flags.
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        
        # Estimate scaling factors for both models using a subset of the token data.
        # This uses only 'n_batches_for_norm_estimate' (default 100) batches,
        # not the entire (possibly huge) dataset.
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
            [estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B],
            device=cfg["device"],
            dtype=torch.float32,
        )
        
        # Optionally refresh (populate) the buffer if desired.
        if refresh_buffer:
            self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        """
        Estimate the norm scaling factor for a given model over a subset of token batches.
        
        For each batch, the L2 norm (averaged over positions and examples) is computed;
        the scaling factor is then determined so that:
            scaling_factor = sqrt(d_model) / mean(norm)
        This ensures that on average the scaled activations have a norm ~√(d_model).
        
        Only n_batches_for_norm_estimate batches are used for efficiency.
        """
        norms_per_batch = []
        for i in tqdm.tqdm(range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"):
            # Select a batch of tokens from all_tokens.
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # Compute the average norm over the activation dimensions.
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm
        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        """
        Populate the activation buffer by running several batches of tokens through the two models.
        
        The activations are:
        - Stacked from model_A and model_B.
        - Rearranged so that each activation vector is a separate row.
        - Written into the preallocated buffer.
        
        Finally, the buffer is randomly permuted to avoid ordering bias.
        """
        self.pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            # Use all available batches on first refresh; later, use only half.
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[
                    self.token_pointer : min(self.token_pointer + self.cfg["model_batch_size"], num_batches)
                ]
                _, cache_A = self.model_A.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                _, cache_B = self.model_B.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                # Stack activations from both models; expected shape: [2, batch, seq_len, d_model].
                acts = torch.stack([cache_A[self.cfg["hook_point"]], cache_B[self.cfg["hook_point"]]], dim=0)
                
                # We no longer drop the BOS token.
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1], self.model_A.cfg.d_model)
                
                # Rearrange from [2, batch, seq_len, d_model] to [(batch * seq_len), 2, d_model].
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )
                # Write these activations into the buffer.
                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
        # Reset pointer and shuffle the buffer to randomize order.
        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        """
        Retrieve the next batch of activations from the buffer.
        
        If the pointer reaches half of the buffer capacity, automatically refresh the buffer.
        Optionally scale (normalize) the activations by the precomputed scaling factors.
        """
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        self.pointer += self.cfg["batch_size"]
        # If more than half the buffer is consumed, refresh the buffer.
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            # Multiply by the scaling factors for each model.
            out = out * self.normalisation_factor[None, :, None]
        return out
