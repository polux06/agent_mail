import torch
from torch import nn
from transformers import GPT2Model, GPT2Config

class BougieGPT(nn.Module):
    """Minimal GPT-style model for candle tokens with asset embeddings."""

    def __init__(self, vocab_size: int, n_assets: int, hidden_size: int = 768,
                 n_layers: int = 12, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=hidden_size,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        self.asset_emb = nn.Embedding(n_assets, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, asset_ids: torch.Tensor):
        asset_embeddings = self.asset_emb(asset_ids).unsqueeze(1)
        inputs_embeds = self.transformer.wte(input_ids) + asset_embeddings
        outputs = self.transformer(inputs_embeds=inputs_embeds)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits
