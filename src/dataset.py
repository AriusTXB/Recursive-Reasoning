import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class SudokuDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Path to 'data/processed'
            split: 'train' or 'test'
        """
        self.split_dir = os.path.join(data_dir, split)
        
        # Load metadata to verify integrity
        meta_path = os.path.join(self.split_dir, "dataset.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Run preprocess first.")
            
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Load arrays
        # We use mmap_mode='r' to keep memory usage low for large datasets
        self.inputs = np.load(os.path.join(self.split_dir, "all__inputs.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(self.split_dir, "all__labels.npy"), mmap_mode='r')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Data is stored as int 1..10 (where 1 is empty in your logic? 
        # Wait, let's stick to your preprocess logic:
        # Original: 0=Empty, 1-9=Digits.
        # Preprocess: arr + 1. So 1=Empty, 2-10=Digits.
        
        input_seq = self.inputs[idx].copy()
        label_seq = self.labels[idx].copy()

        # Convert to LongTensor
        x = torch.from_numpy(input_seq).long()
        y = torch.from_numpy(label_seq).long()
        
        return x, y