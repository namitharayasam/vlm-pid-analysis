# vlm-pid-analysis/models/smolvlm.py

import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from tqdm.auto import tqdm
import os
import pandas as pd

# Relative imports from your project structure
from .base import BaseVLM
from pid.ipfp import convert_data_to_distribution
from pid.metrics import get_measure
from pid.utils import clustering, cluster_embeddings

class SmolVLM(BaseVLM):
    MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"

    TEXT_SLICE_INDEX = 1067 

    def __init__(self, device, **config):
        self.device = device
        self.config = config
        self.processor = None
        self.model = None
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        
        if self.device == "cuda" and self.config.get('quantization', True):
            print("Loading SmolVLM with 4-bit quantization on CUDA.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_NAME, quantization_config=bnb_config, device_map="auto",
                torch_dtype=torch.float16, trust_remote_code=True
            )
        else:
            print(f"Loading SmolVLM on {self.device} without quantization.")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.MODEL_NAME, device_map=self.device, 
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16, 
                trust_remote_code=True
            )
        # Ensure self.device is updated based on model loading
        self.device = self.model.device

    def get_feature_activations(self, batch):
        """Extracts the three embeddings (Text, Vision, Output) for a single batch item."""
        
        # Unpack batch and generate prompt
        messages = batch[0]["messages"]
        image = batch[0]["image"]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process inputs
        processor_outputs = self.processor(
            text=prompt, images=image, return_tensors="pt", padding=True
        )

        input_ids = processor_outputs["input_ids"].to(self.device)
        attention_mask = processor_outputs["attention_mask"].to(self.device)
        pixel_values = processor_outputs["pixel_values"].to(self.device)
        
        # Text Embeddings (Input X2)
        input_ids_pure_text = input_ids[:, self.TEXT_SLICE_INDEX:]
        if input_ids_pure_text.shape[1] == 0:
             return None, None, None # Skip if no text tokens after slicing

        text_embs = self.model.get_input_embeddings()(input_ids_pure_text).squeeze(0)
        
        # Vision Embeddings (Input X1)
        pixel_attention_mask = processor_outputs.get("pixel_attention_mask")
        vision_embs = self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask.to(self.device) if pixel_attention_mask is not None else None
        )
        # Handle the 3D tensor to 2D tensor reduction (mean across patch dimension)
        if len(vision_embs.shape) == 3:
            vision_embs = vision_embs.mean(dim=0)

        # Output Hidden States (Output Y)
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
        last_hidden = output.hidden_states[-1].squeeze(0)

        return text_embs, vision_embs, last_hidden

    def run_ipfp_analysis(self, dataloader, run_config, results_dir="results/smolvlm/gqa"):
        self.model.eval()
        all_profiles = []
        
        # Use config for hyperparameters
        start_index = run_config.get('start_index', 0)
        num_samples = run_config.get('num_samples', len(dataloader))
        end_index = start_index + num_samples
        n_clusters = run_config.get('n_clusters', 10)
        ipfp_iters = run_config.get('ipfp_max_iters', 100)
        pca_components = run_config.get('pca_components', 10)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=end_index):
                if i < start_index: continue
                if i >= end_index: break

                text_embs, vision_embs, last_hidden = self.get_feature_activations(batch)
                
                if text_embs is None: continue # Skip if no valid text tokens

                # Alignment (Clustering to match sequence lengths)
                text_len = text_embs.shape[0]
                image_len = vision_embs.shape[0]
                
                target_len = min(text_len, image_len)
                
                # Align all three tensors to the minimum length (target_len)
                text_embs = cluster_embeddings(text_embs, target_len)
                vision_embs = cluster_embeddings(vision_embs, target_len)
                last_hidden = cluster_embeddings(last_hidden, target_len)
                
                # Clustering and PID
                # Note: n_components is capped inside clustering()
                kmeans_im, _ = clustering(vision_embs, pca=True, n_clusters=n_clusters, n_components=pca_components)
                kmeans_txt, _ = clustering(text_embs, pca=True, n_clusters=n_clusters, n_components=pca_components)
                kmeans_out, _ = clustering(last_hidden, pca=True, n_clusters=n_clusters, n_components=pca_components)

                # P: Joint Distribution P(X1, X2, Y) where X1=Image, X2=Text, Y=Output
                P, _ = convert_data_to_distribution(kmeans_im, kmeans_txt, kmeans_out)
                
                # Calculate PID measures using the IPFP distribution Q
                profile = get_measure(P, name='ipfp', max_iters=ipfp_iters)
                
                profile_meta = {
                    'sample_index': i,
                    'redundancy': float(profile.get('redundancy', 0.0)),
                    'unique_text': float(profile.get('unique1', 0.0)),
                    'unique_image': float(profile.get('unique2', 0.0)),
                    'synergy': float(profile.get('synergy', 0.0)),
                }
                all_profiles.append(profile_meta)
        
        # Save Results
        df_profiles = pd.DataFrame(all_profiles)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'smolvlm_gqa_pid_results_{start_index}_N{num_samples}.csv')
        df_profiles.to_csv(save_path, index=False)

        print(f"\nAnalysis complete. Results saved to {save_path}")
        print("\nMean PID profile:")
        print(df_profiles[['redundancy', 'unique_text', 'unique_image', 'synergy']].mean())
        
        return all_profiles