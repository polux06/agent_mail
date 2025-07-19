import random
import numpy as np
import gymnasium as gym
from typing import Dict, List

from bougie_tokenizer import CandleTokenizer

class MarketEnv(gym.Env):
    """Sequential environment providing rewards for candle prediction."""

    def __init__(self, token_sequences: Dict[int, List[int]], tokenizer: CandleTokenizer, context_len: int = 10):
        super().__init__()
        self.token_sequences = token_sequences
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.asset_ids = list(token_sequences.keys())
        self.action_space = gym.spaces.Discrete(len(tokenizer.codebook))
        self.observation_space = gym.spaces.MultiDiscrete([len(tokenizer.codebook)] * context_len)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.asset_id = random.choice(self.asset_ids)
        self.seq = self.token_sequences[self.asset_id]
        self.pos = 0
        self.done = False
        return np.array(self.seq[: self.context_len], dtype=np.int64), {}

    def step(self, action: int):
        target = self.seq[self.pos + self.context_len]
        pred_candle = self.tokenizer.decode(action)
        true_candle = self.tokenizer.decode(target)
        reward = -float(np.mean(np.abs(np.array(pred_candle) - np.array(true_candle))))
        self.pos += 1
        if self.pos + self.context_len >= len(self.seq) - 1:
            self.done = True
        obs = np.array(self.seq[self.pos : self.pos + self.context_len], dtype=np.int64)
        info = {"asset_id": self.asset_id}
        return obs, reward, self.done, False, info
