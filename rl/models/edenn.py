import math
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
	features_pre: int
	n_flatten: Optional[int] = None
	layer1_out: Optional[torch.Tensor] = None
	layer2_out: Optional[torch.Tensor] = None
	layer3_out: Optional[torch.Tensor] = None
	layer4_out: Optional[torch.Tensor] = None
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int, features_pre: int = 256):
		super().__init__(observation_space, features_dim)
		self.features_pre = features_pre
		self.observation_space = observation_space
		self.get_layers(observation_space.shape[0])

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

		self.layer1 = torch.nn.Sequential(
			conv(n_input_channels, 16, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=16),
		)
		self.layer2 = torch.nn.Sequential(
			conv(16, 32, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=32),
		)
		self.layer3 = torch.nn.Sequential(
			conv(32, 64, kernel_size=kernel_size, stride=stride, bias=True, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm3d(num_features=64),
		)
		self.layer4 = torch.nn.Sequential(
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			conv(64, 64, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
			# conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
		)

		# Compute shape by doing one forward pass
		self.reset_env()
		with torch.no_grad():
			observation = torch.as_tensor(self.observation_space.sample()[None]).float()
			result = self(observation, calc_n_flatten=True)
			# self.n_flatten = result.shape[1]
			self.n_flatten = math.prod(result.shape)

		self.layer5 = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(self.n_flatten, self.features_pre), # 15360 -> ...
			torch.nn.ReLU(inplace=True),
			# torch.nn.Sigmoid(),
			torch.nn.Linear(self.features_pre, self.features_dim),
			# torch.nn.CELU(inplace=True),
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, calc_n_flatten: bool = False) -> torch.Tensor:
		if mask is None:
			mask = (x != 0).float()

		# for i, layer in enumerate(self.layers):
		# 	for sublayer in layer:
		# 		x, mask = self.process(sublayer, x, mask, self.out_c1)
		# 	exec(f"self.out_c{i} = x.detach()")
		for sublayer in self.layer1:
			x, mask = self.process(sublayer, x, mask, self.layer1_out)
		self.layer1_out = x.detach()
		for sublayer in self.layer2:
			x, mask = self.process(sublayer, x, mask, self.layer2_out)
		self.layer2_out = x.detach()
		for sublayer in self.layer3:
			x, mask = self.process(sublayer, x, mask, self.layer3_out)
		self.layer3_out = x.detach()
		for sublayer in self.layer4:
			x, mask = self.process(sublayer, x, mask, self.layer4_out)
		self.layer4_out = x.detach()

		x = x[:, :, -1]
		if calc_n_flatten:
			return x
		# The sizes of `x` and `mask` will diverge here, but that's OK as we don't need the mask anymore
		# We only care about the final bin prediction for now...
		x = self.layer5(x)

		return x

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
		self.layer1_out = None
		self.layer2_out = None
		self.layer3_out = None
		self.layer4_out = None


# ==================================================================================================
class EDeNNPH(EDeNN):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: gym.spaces.Box, features_dim: int, features_pre: int = 256):
		super().__init__(observation_space, features_dim, features_pre)
		self.layer5 = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(self.n_flatten, self.features_pre), # 15360 -> ...
			torch.nn.ReLU(inplace=True),
			# torch.nn.Sigmoid(),
		)
		self.final = torch.nn.Sequential(
			torch.nn.Linear(self.features_pre, self.features_dim),
			# torch.nn.CELU(inplace=True),
		)

	# ----------------------------------------------------------------------------------------------
	def forward_final(self, x: torch.Tensor) -> torch.Tensor:
		x = self.final(x)

		return x
