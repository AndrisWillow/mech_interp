BASE_MODEL_NAME = "Qwen/Qwen2-0.5B" # We are using hooked models - https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
COMPARABLE_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_HOOKPOINT = "blocks.20.hook_resid_pre"
SAE_EXPANSION_RATE = 16
WANDB_PROJECT = "Crosscoders"
WANDB_ENTITY = "andris-willow-"
D_MODEL = 512 # for specific model can be found https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

#Dataset
HF_DS_NAME = "Pile-Lmsys-1m-tokenized-1024-Qwen2.5" # Expecting pretoekized dataset
HF_PROFILE_NAME = "AndrisWillow"

# For loading in a trained crosscoder
HF_CROSSCODER_REPO = "AndrisWillow/Qwen2.5-0.5B-crosscoder-20resid_pre"
HF_CROSSCODER_CONFIG_PATH = "3_cfg.json"
HF_CROSSCODER_WEIGHTS = "3.pt"

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9, # Default
    "beta2": 0.999, # defualt
    "d_in": D_MODEL,
    "dict_size": D_MODEL * SAE_EXPANSION_RATE,
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
