# Mechanistic interpretability
A study project   
Tested using Python 3.12.3   

### Create venv  

python -m venv .venv    
source .venv/bin/activate    
pip install -r requirements.txt     

### login to hugging face   
huggingface-cli login  

### login to wanb  
wandb login   

### Repository guide

Run train.py to train L1 crosscoder    

Almost all vriables are in constants.py  

Use 1_analysis_hist.ipynb to get decoder norm and cosine hisograms, as well as model specific latent IDs  

3_analysis_vis.py - To use latent dashboard and get optional metrics  
Currently it's quite hacky to get latent dashboards as the SAE vis fork has an outdated TransformerLens verion  
You have to esentially download the repo mid script and then run the SAE vis, maybe in the future I will fix the fork and make my own