#!/usr/bin/env python3
"""Compare advantage function between mine and SB3"""

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.ppo.policies import MlpPolicy
from rich import print, inspect
import torch
import numpy as np

env = gym.make("CartPole-v1")
# model = PPO(MlpPolicy, env, verbose=1)
buffer = RolloutBuffer(
	buffer_size=2048,
	observation_space=env.observation_space,
	action_space=env.action_space,
	device="cpu",
	# gae_lambda=0.95,
	gae_lambda=1.0,
	gamma=0.99,
	n_envs=1,
)

last_values = torch.rand([1])
dones = np.array([0])
buffer.compute_returns_and_advantage(last_values, dones)
print(f"returns: {buffer.returns}")
print(f"advantages: {buffer.advantages}")
print(f"values: {buffer.values}")
