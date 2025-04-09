# %%
from transformer_lens import HookedTransformer
from utils import load_pile_lmsys_mixed_tokens, arg_parse_update_cfg
from trainer import Trainer
# %%

# TODO add all variables here defined via CLI
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
COMPARABLE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HF_DS_NAME = "Pile-Lmsys-1m-tokenized-1024-Qwen2.5" # Expecting pretoekized dataset
HF_REPO_NAME = "AndrisWillow"
MODEL_HOOKPOINT = "blocks.13.hook_resid_pre"
SAE_EXPANSION_RATE = 16

WANDB_PROJECT = "Qwen-crosscoders"
WANDB_ENTITY = "andris-willow-"

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
all_tokens = load_pile_lmsys_mixed_tokens()

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9, # Default
    "beta2": 0.999, # defualt
    "d_in": base_model.cfg.d_model,
    "dict_size": base_model.cfg.d_model * SAE_EXPANSION_RATE,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": BASE_MODEL_NAME,
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4, 
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": MODEL_HOOKPOINT,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()

# %%