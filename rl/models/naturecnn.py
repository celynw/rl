"""
Based on NatureCNN from stable-baselines3==1.6.2
Changed colours to increase contrast.
"""
import argparse

from stable_baselines3.common.torch_layers import NatureCNN as SB3_NatureCNN
import gym.spaces
from torch import nn
import torch as th

# ==================================================================================================
class NatureCNN(SB3_NatureCNN):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
		"""
		CNN from DQN nature paper:
			Mnih, Volodymyr, et al.
			"Human-level control through deep reinforcement learning."
			Nature 518.7540 (2015): 529-533.
		Subclassed to remove assertion on observation space.

		Args:
			observation_space (gym.spaces.Box): Observation space.
			features_dim (int, optional): Number of features extracted. This corresponds to the number of unit for the last layer. Defaults to 512.
		"""
		super(SB3_NatureCNN, self).__init__(observation_space, features_dim) # Grandparent class
		# We assume CxHxW images (channels first)
		# Re-ordering will be done by pre-preprocessing or wrapper
		n_input_channels = observation_space.shape[0]
		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
		)

		# Compute shape by doing one forward pass
		with th.no_grad():
			n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Skeleton added for compatibility.

		Args:
			parser (argparse.ArgumentParser): Main parser object.

		Returns:
			argparse.ArgumentParser: Original parser object.
		"""
		return parser
