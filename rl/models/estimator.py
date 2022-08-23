import torch
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==================================================================================================
class Estimator(BaseFeaturesExtractor):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 4):
		super().__init__(observation_space, features_dim)
		# n_input_channels = observation_space.shape[0]
		# self.layers = self.get_layers(n_input_channels)
		# self.final = None
		# self.get_layers(n_input_channels)

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(2, 16, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=16),
		)
		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=32),
		)
		self.layer3 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=64),
		)
		self.layer4 = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(15360, 256),
			# # torch.nn.Dropout(0.5),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 4),
			# # torch.nn.Dropout(0.5),
			# torch.nn.ReLU # TODO
		)
		self.layers = torch.nn.Sequential(
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4,
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		# print(f"x0: {x.shape}")
		# x = self.layer1(x)
		# print(f"x1: {x.shape}")
		# x = self.layer2(x)
		# print(f"x2: {x.shape}")
		# x = self.layer3(x)
		# print(f"x3: {x.shape}")
		# x = self.layer4(x)
		# print(f"x4: {x.shape}")
		x = self.layers(x)

		return x

	# ----------------------------------------------------------------------------------------------
	def reset_env(self):
		pass # Compatibility