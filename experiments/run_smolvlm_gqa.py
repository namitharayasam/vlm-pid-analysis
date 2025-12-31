# vlm-pid-analysis/experiments/run_smolvlm_gqa.py

import torch
import yaml
import os
import random
import numpy as np
from huggingface_hub import login

# Set fixed seed for reproducibility across runs
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# from huggingface_hub import login
login(token="your_token") 

# Imports from your new package structure
from models.smolvlm import SmolVLM
from vlm_data_utils import get_gqa_dataloader

# Define paths
CONFIG_PATH = "configs/smolvlm_gqa.yaml" 
BASE_RESULTS_DIR = "results"

def main():
    # 1. Load Configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Analysis: SmolVLM on GQA (Device: {device}) ---")

    # 2. Initialize Model (This also loads processor and model weights)
    smolvlm = SmolVLM(device=device, **config.get('model', {}))
    
    # 3. Load Data
    dataloader = get_gqa_dataloader(**config.get('dataloader', {}))

    # 4. Run Analysis
    results_dir = os.path.join(BASE_RESULTS_DIR, config['run']['results_sub_dir'])
    smolvlm.run_ipfp_analysis(dataloader, config['run'], results_dir)

    print("--- Analysis Finished ---")

if __name__ == "__main__":
    main()