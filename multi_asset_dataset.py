from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

class MultiAssetDataset(Dataset):
    """Sliding window dataset over multiple assets."""

    def __init__(self, data: Dict[int, List[int]], seq_len: int):
        self.samples: List[Tuple[List[int], int, int]] = []
        self.seq_len = seq_len
        for asset_id, tokens in data.items():
            for i in range(len(tokens) - seq_len):
                context = tokens[i:i + seq_len]
                target = tokens[i + seq_len]
                self.samples.append((context, asset_id, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, asset_id, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(asset_id, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )
