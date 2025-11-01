"""
Dataset and data loading utilities for addition task.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AdditionDataset(Dataset):
    """
    Dataset for multi-digit addition task.

    Loads pre-generated addition problems from JSON files.
    """

    def __init__(self, data_path, pad_token=11):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file
            pad_token: Token used for padding
        """
        self.pad_token = pad_token

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Determine max lengths for padding
        self.max_input_len = max(len(s['input']) for s in self.data)
        self.max_target_len = max(len(s['target']) for s in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict containing:
                - input: Input sequence tensor
                - target: Target sequence tensor
                - input_len: Original input length
                - target_len: Original target length
        """
        sample = self.data[idx]

        # Convert to tensors
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)

        # Original lengths
        input_len = len(input_seq)
        target_len = len(target_seq)

        # Pad sequences
        input_padded = torch.nn.functional.pad(
            input_seq, (0, self.max_input_len - input_len), value=self.pad_token
        )
        target_padded = torch.nn.functional.pad(
            target_seq, (0, self.max_target_len - target_len), value=self.pad_token
        )

        return {
            'input': input_padded,
            'target': target_padded,
            'input_len': input_len,
            'target_len': target_len
        }


def collate_fn(batch):
    """
    Custom collate function for batching.
    """
    inputs = torch.stack([b['input'] for b in batch])
    targets = torch.stack([b['target'] for b in batch])
    input_lens = torch.tensor([b['input_len'] for b in batch])
    target_lens = torch.tensor([b['target_len'] for b in batch])
    return {
        'input': inputs,
        'target': targets,
        'input_len': input_lens,
        'target_len': target_lens
    }


def create_dataloaders(data_dir, batch_size=64, num_workers=0):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing train.json, val.json, test.json
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    train_ds = AdditionDataset(data_dir / 'train.json')
    val_ds = AdditionDataset(data_dir / 'val.json')
    test_ds = AdditionDataset(data_dir / 'test.json')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_vocab_size(num_digits=3):
    """
    Calculate vocabulary size for addition task.
    Returns 12 = 0-9 + '+' + pad
    """
    return 12

