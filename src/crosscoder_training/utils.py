# %%
import plotly.io as pio
pio.renderers.default = "jupyterlab"
import json
import argparse
from datasets import load_dataset
from pathlib import Path
import torch
from IPython import get_ipython
from train import HF_DS_NAME, HF_REPO_NAME

# crosscoder stuff

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg    

def load_pile_lmsys_mixed_tokens():
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "workspace" / "data"
    cache_dir = script_dir / "workspace" / "cache"

    data_file = data_dir / f"{HF_DS_NAME}.pt"
    hf_disk_dir = data_dir / f"{HF_DS_NAME}.hf"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading data from disk")
        all_tokens = torch.load(data_file)
    except:
        print("Data is not cached. Loading data from HF")
        data = load_dataset(
            f"{HF_REPO_NAME}/{HF_DS_NAME}", 
            split="train", 
            cache_dir=str(cache_dir)
        )
        data.save_to_disk(str(hf_disk_dir))
        data.set_format(type="torch", columns=["input_ids"])
        all_tokens = data["input_ids"]
        torch.save(all_tokens, data_file)
        print(f"Saved tokens to disk at {data_file}")
    return all_tokens
