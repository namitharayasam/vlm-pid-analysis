# vlm-pid-analysis/datasets/base.py

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader

class BaseVLDataset(Dataset, ABC):
    """
    Abstract Base Class for all Vision-Language Datasets.
    
    All concrete dataset implementations (GQADataset, VQADataset, etc.) 
    must inherit from this class and implement the abstract methods.
    """

    def __init__(self, data):
        """Initializes the dataset with loaded raw data."""
        self.data = data
        self.dataset_name = "BaseDataset"

    @abstractmethod
    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Retrieves one item from the dataset.
        
        The returned dictionary should contain:
        - 'image': PIL Image object
        - 'question': str
        - 'answer': str (ground truth)
        - 'messages': list (The chat template structure for the VLM)
        """
        pass

def get_base_dataloader(dataset_class, batch_size=1, shuffle=False, **kwargs):
    """
    A generic function to create a DataLoader for any class inheriting from BaseVLDataset.
    
    Note: Your analysis uses collate_fn=lambda x: x for single-item batches.
    """
    dataset = dataset_class(**kwargs)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda x: x, 
        **kwargs
    )