"""Example PPO training loop for BougieGPT using stable-baselines3."""
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from typing import Any

from gpt_bougie_model import BougieGPT
from market_env import MarketEnv

class GPTExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, vocab_size: int, n_assets: int, hidden_size: int = 768):
        super().__init__(observation_space, features_dim=hidden_size)
        self.model = BougieGPT(vocab_size, n_assets, hidden_size=hidden_size)
        self.features_dim = hidden_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = observations.long()
        batch_size = tokens.shape[0]
        asset_ids = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        outputs = self.model(tokens, asset_ids)
        return outputs[:, -1, :]

class GPTPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, vocab_size: int, n_assets: int, **kwargs):
        self.extractor = GPTExtractor(observation_space, vocab_size, n_assets)
        super().__init__(observation_space, action_space, lr_schedule,
                         features_extractor=self.extractor,
                         features_extractor_class=None,
                         features_extractor_kwargs=None,
                         **kwargs)


def train(env: MarketEnv, timesteps: int = 1000):
    policy_kwargs = dict(vocab_size=len(env.tokenizer.codebook), n_assets=len(env.asset_ids))
    model = PPO(GPTPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

if __name__ == "__main__":
    print("This module provides the train() function for PPO training.")
