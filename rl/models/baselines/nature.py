"""
Based on NatureCNN from stable-baselines3==1.6.0
Changed colours to increase contrast.
"""
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym.spaces
from torch import nn
import torch as th

# ==================================================================================================
class NatureCNN(BaseFeaturesExtractor):
	"""
	CNN from DQN nature paper:
		Mnih, Volodymyr, et al.
		"Human-level control through deep reinforcement learning."
		Nature 518.7540 (2015): 529-533.

	:param observation_space:
	:param features_dim: Number of features extracted.
		This corresponds to the number of unit for the last layer.
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_pre: int, features_dim: int = 512, no_pre: bool = False):
		# features_pre is for compatibility...
		# no_pre is for compatibility...
		super().__init__(observation_space, features_dim)
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
	def forward(self, observations: th.Tensor) -> th.Tensor:
		return self.linear(self.cnn(observations))

	# ----------------------------------------------------------------------------------------------
	def reset_env(self):
		pass # Compatibility
