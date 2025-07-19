# Crypto Candle GPT Prototype

This repository contains a minimal implementation of a GPT-style model that works on
quantized OHLCV candles. It includes utilities for tokenising candles, building a
multi-asset dataset and running supervised or reinforcement learning training loops.

## Modules

- `bougie_tokenizer.py` – discretises and encodes full candles into integer tokens
- `gpt_bougie_model.py` – GPT model with an additional asset embedding
- `multi_asset_dataset.py` – sliding window dataset over token sequences
- `market_env.py` – simple market environment producing rewards for predictions
- `train_supervised.py` – example cross-entropy pretraining loop
- `rl_trainer.py` – skeleton PPO training routine using stable-baselines3
