# L1 Crosscoders

This repository contains scripts to train a L1 crosscoder, prepare the necessary datasets, perform analysis and create feature dashboards.

This repository could be considered a fork from https://github.com/ckkissane/crosscoder-model-diff-replication
It contains a lot of the original code and ideas, but is heavily modified to include some data preparing scripts, better visulaization, documentation and refactoring.

## Repository guide
* `tokenize_datasets.ipynb` pulls lmsys and pile datasets and then pretokenizes them for the relavant LLM. You can modify to include chat template or other specific model tokens.

### corsscoder
* `train.py` run to train L1 crosscoder.
* `trainer.py` contains the pytorch boilerplate code that actually trains the crosscoder.
* `crosscoder.py` contains the pytorch implementation of the crosscoder.
* `buffer.py` contains code to extract activations from both models, concatenate them, and store them in a buffer which is shuffled and periodically refreshed during training.    
* `constants.py` contains all relavant variables to change for training or other scripts.
* `upload_to_hf.py` uploads the specified crosscoder model weights and config to hugging face.

#### After training

* `1_analysis_hist.ipynb` to get decoder norm and cosine similarity hisograms, as well as model specific latent IDs.
* `2_analysis_vis.py` to use latent dashboard and get optional metrics.

#### feature_dashboards
This directory contains feature dashboards from the trained crosscoders. It shows activations for sampled instruct, base specific and shared latents.

## Useful scripts for setup:

### Create venv  
python -m venv .venv    
source .venv/bin/activate    
pip install -r requirements.txt     

### login to hugging face   
huggingface-cli login  

### login to wanb  
wandb login