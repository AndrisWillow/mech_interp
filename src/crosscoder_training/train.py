# %%
from transformer_lens import HookedTransformer
from utils import load_HF_tokenized_DS, arg_parse_update_cfg
from trainer import Trainer
from constants import BASE_MODEL_NAME, COMPARABLE_MODEL_NAME, default_cfg
# %%
# Load models
device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    BASE_MODEL_NAME, 
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    COMPARABLE_MODEL_NAME, 
    device=device, 
)

# %%
all_tokens = load_HF_tokenized_DS()

# %%
cfg = arg_parse_update_cfg(default_cfg) # TODO see if we still need this

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()

# %%