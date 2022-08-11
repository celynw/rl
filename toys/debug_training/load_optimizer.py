#!/usr/bin/env python3
import colored_traceback.auto
from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain

import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from rich import print, inspect

import rl

# ==================================================================================================
class Estimator(BaseFeaturesExtractor):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 4):
		super().__init__(observation_space, features_dim)

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(2, 16, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			torch.nn.GroupNorm(num_groups=4, num_channels=16),
		)
		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			torch.nn.GroupNorm(num_groups=4, num_channels=32),
		)
		self.layer3 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			torch.nn.GroupNorm(num_groups=4, num_channels=64),
		)
		self.layer4 = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(15360, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 4),
		)
		self.layers = torch.nn.Sequential(
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4,
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		return self.layers(x)

	# ----------------------------------------------------------------------------------------------
	def reset_env(self):
		pass # Compatibility


# ==================================================================================================
def main():
	env_ = gym.make("CartPole-events-v1")
	# env_ = gym.make("CartPole-events-debug")
	# env_ = gym.make("CartPole-v1")

	env = Monitor(env_, "/tmp/rl", allow_early_resets=True)
	policy_kwargs = dict(features_extractor_class=Estimator)
	model = PPO(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, device="cpu", n_steps=16) # n_steps=2048, total_timesteps will be at least this
	env.set_model(model.policy.features_extractor)
	checkpoint = torch.load("/home/cw0071/dev/python/rl/train_estimator_RL_estimator/219_1e4b47kx/checkpoints/epoch=59-step=75000.ckpt")

	# DEBUG
	print(checkpoint["optimizer_states"][0]["state"][0].keys())
	print(f'model params: {len(model.policy.optimizer.param_groups[0]["params"])}')
	print(f'checkpoint params: {len(checkpoint["optimizer_states"][0]["state"])}')
	model.policy.features_extractor.load_state_dict(checkpoint["state_dict"])
	# model.policy.optimizer.load_state_dict(checkpoint["optimizer_states"][0])

	model.learn(total_timesteps=16, callback=[TqdmCallback()])

	# oldParams = model.policy.optimizer.param_groups[0]["params"]
	oldParams = model.policy.optimizer.state
	load_state_dict(model.policy.optimizer, checkpoint["optimizer_states"][0])
	# newParams = model.policy.optimizer.param_groups[0]["params"]
	newParams = model.policy.optimizer.state
	print(oldParams == newParams)

# ==================================================================================================
def load_state_dict(self, state_dict):
	r"""Loads the optimizer state.

	Args:
		state_dict (dict): optimizer state. Should be an object returned
			from a call to :meth:`state_dict`.
	"""
	# deepcopy, to be consistent with module API
	state_dict = deepcopy(state_dict)
	# Validate the state_dict
	groups = self.param_groups
	saved_groups = state_dict['param_groups']

	if len(groups) != len(saved_groups):
		raise ValueError("loaded state dict has a different number of "
							"parameter groups")
	param_lens = (len(g['params']) for g in groups)
	saved_lens = (len(g['params']) for g in saved_groups)
	if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
		pass
		# raise ValueError("loaded state dict contains a parameter group "
		# 					"that doesn't match the size of optimizer's group")

	# Update the state
	id_map = {old_id: p for old_id, p in
				zip(chain.from_iterable((g['params'] for g in saved_groups)),
					chain.from_iterable((g['params'] for g in groups)))}

	def cast(param, value):
		r"""Make a deep copy of value, casting all tensors to device of param."""
		if isinstance(value, torch.Tensor):
			# Floating-point types are a bit special here. They are the only ones
			# that are assumed to always match the type of params.
			if param.is_floating_point():
				value = value.to(param.dtype)
			value = value.to(param.device)
			return value
		elif isinstance(value, dict):
			return {k: cast(param, v) for k, v in value.items()}
		elif isinstance(value, container_abcs.Iterable):
			return type(value)(cast(param, v) for v in value)
		else:
			return value

	# Copy state assigned to params (and cast tensors to appropriate types).
	# State that is not assigned to params is copied as is (needed for
	# backward compatibility).
	state = defaultdict(dict)
	for k, v in state_dict['state'].items():
		print(f"{k} in map: {k in id_map}")
		if k in id_map:
			param = id_map[k]
			state[param] = cast(param, v)
		else:
			state[k] = v

	# Update parameter groups, setting their 'params' value
	def update_group(group, new_group):
		new_group['params'] = group['params']
		return new_group
	param_groups = [
		update_group(g, ng) for g, ng in zip(groups, saved_groups)]
	self.__setstate__({'state': state, 'param_groups': param_groups})


# ==================================================================================================
class TqdmCallback(BaseCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		super().__init__()
		self.progress_bar = None

	# ----------------------------------------------------------------------------------------------
	def _on_training_start(self):
		# self.progress_bar = tqdm(total=self.locals["total_timesteps"], position=1, leave=False)
		self.progress_bar = tqdm(total=self.locals["total_timesteps"], position=1, leave=True)

	# ----------------------------------------------------------------------------------------------
	def _on_step(self):
		self.progress_bar.update(1)
		return True

	# ----------------------------------------------------------------------------------------------
	def _on_training_end(self):

		self.progress_bar.close()
		self.progress_bar = None


# ==================================================================================================
if __name__ == "__main__":
	main()
