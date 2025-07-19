"""Simple supervised pretraining for BougieGPT."""
import torch
from torch.utils.data import DataLoader
from torch import nn

from gpt_bougie_model import BougieGPT
from multi_asset_dataset import MultiAssetDataset


def train(dataset: MultiAssetDataset, vocab_size: int, n_assets: int, epochs: int = 1, batch_size: int = 8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BougieGPT(vocab_size=vocab_size, n_assets=n_assets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for contexts, asset_ids, targets in dataloader:
            logits = model(contexts, asset_ids)
            loss = criterion(logits[:, -1, :], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    print("This module provides the train() function for supervised pretraining.")
