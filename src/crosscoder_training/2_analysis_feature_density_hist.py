# %%
from crosscoder import CrossCoder
import plotly.express as px
import torch
from utils import load_HF_tokenized_DS

torch.set_grad_enabled(False); # for memory reduction
# Load the crosscoder and both the base and comparable (chat) models.
cross_coder = CrossCoder.load_from_hf()

# Load tokenized dataset (all_tokens could be very large, but our Buffer uses a subset for scaling).
all_tokens = load_HF_tokenized_DS()

device = 'cuda:0'

## 1. Load in the crosscoder
# 2. Generate activations on dataset
# 3. Plot log10 feature density (x) vs num of features (y)
