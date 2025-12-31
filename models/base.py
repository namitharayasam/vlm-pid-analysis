# vlm-pid-analysis/models/base.py

from abc import ABC, abstractmethod

class BaseVLM(ABC):
    """
    Abstract Base Class for all Vision-Language Models (VLM) 
    used in the PID analysis. 
    
    All concrete model implementations (SmolVLM, LLaVA, etc.) 
    must inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def __init__(self, device, **config):
        """Initializes the model, processor, and loads weights."""
        pass

    @abstractmethod
    def get_feature_activations(self, batch):
        """
        Extracts the three required feature tensors (X1, X2, Y) 
        from a single batch item:
        - X1 (Vision Embeddings)
        - X2 (Text Embeddings)
        - Y (Output Hidden States)
        Returns: text_embs, vision_embs, last_hidden
        """
        pass

    @abstractmethod
    def run_ipfp_analysis(self, dataloader, run_config, results_dir):
        """
        Executes the main PID analysis loop over the dataloader 
        and saves results.
        """
        pass