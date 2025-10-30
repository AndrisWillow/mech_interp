# %%
import plotly.io as pio
pio.renderers.default = "jupyterlab"
import json
import argparse
from datasets import load_dataset, load_from_disk
from pathlib import Path
import torch
from IPython import get_ipython
from constants import HF_DS_NAME, HF_PROFILE_NAME

# TODO maybe remove this function?
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

def load_HF_tokenized_DS(as_tensor=False):
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "workspace" / "data"
    cache_dir = script_dir / "workspace" / "cache"

    hf_disk_dir = data_dir / f"{HF_DS_NAME}.hf"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading dataset from disk")
        dataset = load_from_disk(str(hf_disk_dir))
    except:
        print("Data is not cached. Loading dataset from HF")
        dataset = load_dataset(
            f"{HF_PROFILE_NAME}/{HF_DS_NAME}", 
            split="train", 
            cache_dir=str(cache_dir)
        )
        dataset.save_to_disk(str(hf_disk_dir))
        print(f"Saved dataset to disk at {hf_disk_dir}")

    dataset.set_format(type="torch", columns=["input_ids"])

    if as_tensor:
        # Returning all tokens as a tensor
        return torch.cat([x["input_ids"] for x in dataset])

    return dataset

