from typing import Optional

import torch
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.models.utils import Decay3dPartial

# ==================================================================================================
class EDeNN(BaseFeaturesExtractor):
	"""
	:param observation_space: (gym.Space)
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
		# `features_dim` should be 128 to match self.mid?
		super().__init__(observation_space, features_dim)
		n_input_channels = observation_space.shape[0]
		# print(f"n_input_channels: {n_input_channels}")
		# self.layers = self.get_layers(n_input_channels)
		self.final = None
		self.get_layers(n_input_channels)

		# Compute shape by doing one forward pass
		self.reset_env()
		with torch.no_grad():
			observation = torch.as_tensor(observation_space.sample()[None]).float()
			n_flatten = self(
				observation,
				(observation != 0).float()
			).shape[1]

		self.final = torch.nn.Sequential(
			torch.nn.Linear(n_flatten, 1024), # 15360 -> ...
			torch.nn.ReLU(inplace=True),
			# torch.nn.Sigmoid(),
			torch.nn.Linear(1024, self.features_dim), # ... -> 128
			# torch.nn.CELU(inplace=True),
		)

	# ----------------------------------------------------------------------------------------------
	def process(self, layer: torch.nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None, previous_output: Optional[torch.Tensor] = None):
		if isinstance(layer, torch.nn.Linear):
			x = x.mean(dim=(3, 4)) # NCDHW -> NCD
			x = x.permute(0, 2, 1) # NCD -> NDC
			x = layer(x)
			x = x.permute(0, 2, 1) # NDC -> NCD
		elif isinstance(layer, Decay3dPartial):
			if mask is not None:
				x, mask, weights = layer(x, mask, previous_output)
		else:
			x = layer(x)

		return x, mask

	# ----------------------------------------------------------------------------------------------
	def reset_env(self):
		self.out_c1 = None
		self.out_c2 = None
		self.out_c3 = None
		self.out_c4 = None
		self.out_mid = None

	# ----------------------------------------------------------------------------------------------
	# def forward(self, x: torch.Tensor) -> torch.Tensor:
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# DEBUG
		mask = (x != 0).float()
		# print(f"x, mask shape 1: {x.shape}, {mask.shape}")

		for sublayer in self.conv1:
			x, mask = self.process(sublayer, x, mask, self.out_c1)
		# print(f"x, mask shape 2: {x.shape}, {mask.shape}")
		self.out_c1 = x.detach()

		for sublayer in self.conv2:
			x, mask = self.process(sublayer, x, mask, self.out_c2)
		# print(f"x, mask shape 3: {x.shape}, {mask.shape}")
		self.out_c2 = x.detach()

		for sublayer in self.conv3:
			x, mask = self.process(sublayer, x, mask, self.out_c3)
		# print(f"x, mask shape 4: {x.shape}, {mask.shape}")
		self.out_c3 = x.detach()

		# for sublayer in self.conv4:
		# 	x, mask = self.process(sublayer, x, mask, self.out_c4)
		# # print(f"x, mask shape 5: {x.shape}, {mask.shape}")
		# self.out_c4 = x.detach()

		for sublayer in self.mid:
			x, mask = self.process(sublayer, x, mask, self.out_mid)
		# print(f"x, mask shape 6: {x.shape}, {mask.shape}")
		self.out_mid = x.detach()

		# The sizes of `x` and `mask` will diverge here, but that's OK as we don't need the mask anymore
		# We only care about the final bin prediction for now...
		x = x[:, :, -1]
		# print(f"x shape 8: {x.shape}")

		x = self.flatten(x)
		# print(f"x shape 9: {x.shape}")

		if self.final is not None:
			x = self.final(x)
			# print(f"x shape 10: {x.shape}")


		# quit(0)
		return x

	# ----------------------------------------------------------------------------------------------
	def get_layers(self, n_input_channels: int):
		# D, H, W
		kernel_size = 3
		stride = (1, 2, 2)
		pad = (0, 1, 1)
		scale_factor = (1, 2, 2)

		partial_kwargs = {}
		conv = Decay3dPartial
		partial_kwargs["multi_channel"] = True
		partial_kwargs["return_mask"] = True
		partial_kwargs["kernel_ratio"] = True

		self.conv1 = torch.nn.Sequential(
			conv(n_input_channels, 16, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=16),
		)
		self.conv2 = torch.nn.Sequential(
			conv(16, 32, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=32),
		)
		self.conv3 = torch.nn.Sequential(
			conv(32, 64, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=64),
		)
		# self.conv4 = torch.nn.Sequential(
		# 	conv(64, 128, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, return_decay=True, **partial_kwargs),
		# 	torch.nn.ReLU(inplace=True),
		# 	torch.nn.BatchNorm3d(num_features=128),
		# )

		# Bias?
		self.mid = torch.nn.Sequential(
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			conv(64, 64, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
		)
		self.flatten = torch.nn.Flatten()

		# self.final = torch.nn.Linear(128, 64, bias=True)
		# self.final1 = torch.nn.Linear(18, 1, bias=True)
		# self.final1 = torch.nn.Linear(60, 1, bias=True)

		# return torch.nn.Sequential(
		# 	self.conv1,
		# 	self.conv2,
		# 	self.conv3,
		# 	# self.conv4,
		# 	self.mid,
		# 	# torch.nn.Flatten(),
		# )
