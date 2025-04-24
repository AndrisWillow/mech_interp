# %%
from crosscoder import CrossCoder
import plotly.express as px
import torch
from constants import HF_CROSSCODER_REPO

torch.set_grad_enabled(False); # for memory reduction
# %%
cross_coder = CrossCoder.load_from_hf()

# %%
norms = cross_coder.W_dec.norm(dim=-1)
norms.shape
# %%
relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms.shape
# %%

fig = px.histogram(
    relative_norms.detach().cpu().numpy(),
    title=f"{HF_CROSSCODER_REPO}",
    labels={"value": "Relatīvās dekoderu latentu normu atšķirības"}, #  Decoder latent relative norm difference
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(
    title_text="Latentu skaits", # Latent count
    type="log",
    tickvals=[10**i for i in range(0, 6)],  # 10^0 to 10^5
    ticktext=[f"10^{i}" for i in range(0, 6)]
)

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    ticktext=['0', '0.25', '0.5', '0.75', '1.0']
)

fig.show()

# %%
shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)
shared_latent_mask.shape
# %%
# Cosine similarity of recoder vectors between models

cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1))
cosine_sims.shape
# %%


fig = px.histogram(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(), 
    #title="Cosine similarity of decoder vectors between models",
    log_y=True,  # Sets the y-axis to log scale
    range_x=[-1, 1],  # Sets the x-axis range from -1 to 1
    nbins=100,  # Adjust this value to change the number of bins
    labels={"value": "Cosine similarity of decoder vectors between models"}
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents (log scale)")

fig.show()

# %%
# Extract IT-specific, Base-specific, and Shared latent indices for SAE Vis analysis

# Get latents where the second model (e.g. IT model) dominates
it_specific_latent_ids = torch.where(relative_norms > 0.99)[0]

# Get latents where the first model (e.g. base model) dominates
base_specific_latent_ids = torch.where(relative_norms <= 0.0025)[0]

# Get latents that are "shared" between the models — i.e., near-balanced decoder norm
shared_mask = (relative_norms >= 0.5) & (relative_norms <= 0.55)
shared_latent_ids_all = torch.where(shared_mask)[0]

# Sample 100 random shared latents from the full shared set
num_samples = 100
if len(shared_latent_ids_all) > num_samples:
    sampled_indices = torch.randperm(len(shared_latent_ids_all))[:num_samples]
    shared_latent_ids = shared_latent_ids_all[sampled_indices]
else:
    shared_latent_ids = shared_latent_ids_all  # Use all if fewer than 100 available

# Convert the tensors to plain Python lists for compatibility with plotting, storage, etc.
it_specific_latents = it_specific_latent_ids.tolist()
base_specific_latents = base_specific_latent_ids.tolist()
shared_latents = shared_latent_ids.tolist()

# Logging the results
print(f"Found {len(it_specific_latents)} IT-specific latents")
print("IT-specific latent indices:", it_specific_latents)

print(f"Found {len(base_specific_latents)} Base-specific latents")
print("Base-specific latent indices:", base_specific_latents)

print(f"Sampled {len(shared_latents)} shared latents")
print("Shared latent indices (sampled from relative norm 0.5–0.55):", shared_latents)
# %%
