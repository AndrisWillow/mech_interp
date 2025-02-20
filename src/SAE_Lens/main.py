import torch
import plotly.express as px
from functools import partial
from datasets import load_dataset
from transformer_lens import utils
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
print("PyTorch using GPU:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

PORT = 8000

# BitsAndBytes 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load quantized Gemma-2B model
model_name = "google/gemma-2b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Sparse Autoencoder
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-2b-res-jb",
    sae_id="blocks.0.hook_resid_post",
    device=device,
)
sae.to(device)

# Load dataset
dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = utils.tokenize_and_concatenate(
    dataset=dataset,
    tokenizer=tokenizer,
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

# Test 1:
sae.eval()

with torch.no_grad(), torch.amp.autocast(device_type=device):
    batch_tokens = token_dataset[:16]["tokens"].to(device)
    outputs = model(batch_tokens, output_hidden_states=True)
    cache = outputs.hidden_states

    # Use the SAE
    feature_acts = sae.encode(cache[-1].to(device))
    sae_out = sae.decode(feature_acts)
    
    # Save memory
    del cache

    # Analyze SAE activations
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l0", l0.mean().item())
    px.histogram(l0.flatten().cpu().numpy()).show()

# Reconstruction test
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

print("Orig", model(batch_tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
    ).item(),
)