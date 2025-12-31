# vlm-pid-analysis/vlm_data_utils.py (Final Version)

from torch.utils.data import DataLoader
# Import the external library using an alias
from datasets import load_dataset as hf_load_dataset 
# Import your clean Dataset class from the local package
from data.gqa import GQADataset 

def load_gqa_data(split="testdev_balanced"):
    """
    Loads and combines image and instruction data for GQA, correcting for 
    the subset/split key mismatch in the lmms-lab/GQA dataset.
    """
    
    # 1. Determine the correct internal split key name.
    #    The subset 'testdev_balanced' has an internal split key of 'testdev'.
    #    The subset 'train_balanced' has an internal split key of 'train'.
    #    We use split('_')[0] to get the correct internal split name.
    actual_split_name = split.split('_')[0] 
    
    # 2. Load GQA dataset components using the *full subset name* (e.g., 'testdev_balanced')
    #    The returned object is a DatasetDict containing the actual_split_name as a key.
    images_subset = hf_load_dataset("lmms-lab/GQA", f"{split}_images")
    instructions_subset = hf_load_dataset("lmms-lab/GQA", f"{split}_instructions")

    # 3. FIX: Access the loaded data using the *actual_split_name* (e.g., 'testdev')
    #    This resolves the KeyError you were seeing.
    images_data = images_subset[actual_split_name] 
    instructions_data = instructions_subset[actual_split_name]

    # Map images by ID
    images_by_id = {item['id']: item['image'] for item in images_data}
    combined_data = []

    # Combine instructions with the corresponding image object
    for item in instructions_data:
        image_id = item['imageId']
        if image_id in images_by_id:
            combined_data.append({
                'image': images_by_id[image_id],
                'question': item['question'],
                'answer': item['answer'],
                'fullAnswer': item['fullAnswer'],
            })
    
    return GQADataset(combined_data)

def get_gqa_dataloader(batch_size=1, split="testdev_balanced", shuffle=False, **kwargs):
    """Creates the dataloader for the GQA dataset."""
    dataset = load_gqa_data(split=split)
    # Use the custom collate_fn=lambda x: x for single-item batches
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x, **kwargs)