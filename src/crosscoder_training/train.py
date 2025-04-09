# %%
from utils import *
from trainer import Trainer
# %%
device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-0.5B", 
    device=device, 
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    device=device, 
)

# %%
all_tokens = load_pile_lmsys_mixed_tokens()

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2, # maybe use 4? Seems a little low.
    "beta1": 0.9, # Default
    "beta2": 0.999, # defualt
    "d_in": base_model.cfg.d_model,
    "dict_size": base_model.cfg.d_model * 16, # 896 * 16 = 14,336
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4, 
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.13.hook_resid_pre", # figure out which hookpoint to add crosscoder
    "wandb_project": "Qwen-crosscoders",
    "wandb_entity": "andris-willow-",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()

# %%