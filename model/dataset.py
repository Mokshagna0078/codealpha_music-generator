"""
PyTorch Dataset for music token sequences.
Loads preprocessed token sequences and creates batches for training.
"""
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


class MusicTokenDataset(Dataset):
    """
    Dataset of preprocessed music token sequences.
    Each sequence is a variable-length array of token IDs.
    """
    
    def __init__(self, data_path: str | Path, max_seq_len: int = 2048):
        """
        Args:
            data_path: Path to the .npy file containing token sequences
            max_seq_len: Maximum sequence length (sequences are chunked to this)
        """
        self.max_seq_len = max_seq_len
        
        # Load tokenized sequences
        data = np.load(data_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 1 and data.dtype == object:
            # Array of sequences (each element is a sequence)
            self.sequences = [seq.tolist() if isinstance(seq, np.ndarray) else list(seq) for seq in data]
        else:
            # Single large array - split into sequences by EOS tokens
            self.sequences = self._split_by_eos(data.tolist())
        
        # Create chunks for training
        self.chunks = self._create_chunks()
    
    def _split_by_eos(self, tokens: List[int], eos_token: int = 2) -> List[List[int]]:
        """Split a long token sequence by EOS tokens."""
        sequences = []
        current = []
        for token in tokens:
            current.append(token)
            if token == eos_token and len(current) >= 10:  # Minimum sequence length
                sequences.append(current)
                current = []
        if len(current) > 5:
            sequences.append(current)
        return sequences
    
    def _create_chunks(self) -> List[List[int]]:
        """
        Split sequences into fixed-length chunks for training.
        Overlapping chunks with stride = max_seq_len // 2.
        """
        chunks = []
        stride = self.max_seq_len // 2
        
        for seq in self.sequences:
            if len(seq) <= self.max_seq_len:
                # Pad shorter sequences
                padded = seq + [0] * (self.max_seq_len - len(seq))
                chunks.append(padded)
            else:
                # Create overlapping chunks
                for start in range(0, len(seq) - self.max_seq_len + 1, stride):
                    chunks.append(seq[start:start + self.max_seq_len])
                # Last chunk (may overlap with previous)
                if len(seq) > self.max_seq_len and (len(seq) - self.max_seq_len) % stride != 0:
                    chunks.append(seq[-self.max_seq_len:])
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with 'input_ids' and 'labels' tensors
        """
        tokens = self.chunks[idx]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Input: tokens[:-1], Labels: tokens[1:]
        # For autoregressive training
        input_ids = tokens_tensor[:-1]
        labels = tokens_tensor[1:]
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
    def get_vocab_size(self, tokenizer_path: Optional[str | Path] = None) -> int:
        """
        Get vocabulary size from tokenizer if available.
        Falls back to max token ID + 1 from the data.
        """
        if tokenizer_path and Path(tokenizer_path).exists():
            with open(tokenizer_path) as f:
                data = json.load(f)
            return data.get("vocab_size", 0)
        
        # Fallback: find max token ID
        max_id = 0
        for chunk in self.chunks:
            max_id = max(max_id, max(chunk))
        return max_id + 1


def create_dataloader(
    data_path: str | Path,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Create a DataLoader for music token sequences.
    
    Args:
        data_path: Path to .npy token data file
        batch_size: Number of sequences per batch
        max_seq_len: Maximum sequence length
        shuffle: Whether to shuffle the dataset
        num_workers: Number of dataloader worker processes
    
    Returns:
        DataLoader that yields dicts with 'input_ids', 'labels', 'attention_mask'
    """
    dataset = MusicTokenDataset(data_path, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for consistent training
    )
    return dataloader
