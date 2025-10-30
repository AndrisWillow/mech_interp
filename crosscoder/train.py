# %%
from transformer_lens import HookedTransformer
from utils import load_HF_tokenized_DS, arg_parse_update_cfg
from trainer import Trainer
from constants import default_cfg
# %%
# Load models
device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    default_cfg["base_model"], 
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    default_cfg["comperable_model"], 
    device=device, 
)

# %%
all_tokens = load_HF_tokenized_DS(as_tensor=True)

# %%
cfg = arg_parse_update_cfg(default_cfg) # TODO see if we still need this

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()

# %%