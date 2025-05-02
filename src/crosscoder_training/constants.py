BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B" # We are using hooked models - https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
COMPARABLE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_HOOKPOINT = "blocks.13.hook_resid_pre"
SAE_EXPANSION_RATE = 16
WANDB_PROJECT = "Crosscoders"
WANDB_ENTITY = "andris-willow-"
D_MODEL = 512 # for specific model can be found https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

#Dataset
HF_DS_NAME = "Pile-Lmsys-1m-tokenized-1024-Qwen2.5-WTemplTok" # Expecting pretoekized dataset
HF_PROFILE_NAME = "AndrisWillow"

# For loading in a trained crosscoder
HF_CROSSCODER_REPO = "AndrisWillow/Qwen2.5-0.5B-0.5B_it-cc_diff_WTemplTok-13l-resid_pre-14k"  # "AndrisWillow/Qwen2.5-0.5B-crosscoder-13resid_pre"
HF_CROSSCODER_CONFIG_PATH = "cfg.json"
HF_CROSSCODER_WEIGHTS = "cc_weights.pt"

default_cfg = {
    "seed": 49,
    "seq_len": 1024, # Maximum context length the model is trained on; More would be better
    "batch_size": 4096, # Larger size makes for better gradient stability, lower gives more frequent gradient updates, but might decrease training stability
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2, # L1 sparsity coef
    "beta1": 0.9, # Default; Adam optimizer beta1
    "beta2": 0.999, # Defualt; Adam optimizer beta2
    "d_in": D_MODEL, # Crosscoder input dimension
    "dict_size": D_MODEL * SAE_EXPANSION_RATE, # Crosscoder dict dimension
    "enc_dtype": "fp32",
    "base_model": BASE_MODEL_NAME,
    "comperable_model": COMPARABLE_MODEL_NAME,
    "site": "resid_pre",
    "device": "cuda:0",
    "dec_init_norm": 0.08, # TODO explain what it is and what it does; This paramater is quite sensitive and could be explored more what is an optimal value for it
    # WandB settings
    "log_every": 100, # Log to WandB? [Are these steps?]
    "save_every": 30000, # Save every steps? [Log to WandB?]
    "hook_point": MODEL_HOOKPOINT,
    "wandb_project": WANDB_PROJECT, 
    "wandb_entity": WANDB_ENTITY, # Your WandB team name
    # Buffer settings
    "buffer_mult": 128, # TODO add simple explenation 
    "model_batch_size": 4, # Number of token-chunks to process per refresh loop
}
