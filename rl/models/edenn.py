import argparse
import math
from typing import Optional

import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.models.utils import Decay3dPartial

# ==================================================================================================
class EDeNN(BaseFeaturesExtractor):
	layer1_out: Optional[torch.Tensor] = None
	layer2_out: Optional[torch.Tensor] = None
	layer3_out: Optional[torch.Tensor] = None
	layer4_out: Optional[torch.Tensor] = None
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: spaces.Box, features_dim: int, projection_head: bool = True, projection_dim: int = 256):
		"""
		Feature extractor using "EDeNN: Event Decay Neural Networks for low latency vision" (Celyn Walters, Simon Hadfield).
		https://arxiv.org/abs/2209.04362

		Args:
			observation_space (spaces.Box): Observation space from environment.
			features_dim (int): Output size of final layer (features) for RL policy.
			projection_head (bool, optional): Use projection head or not. Defaults to True.
			projection_dim (int, optional): Output size of projection head layer. Defaults to 256.
		"""
		super().__init__(observation_space, features_dim)
		self.observation_space = observation_space
		self.projection_head = projection_head
		self.projection_dim = projection_dim
		self.init_layers(self.observation_space.shape[0])

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Appends model-specific arguments to the parser.

		Args:
			parser (argparse.ArgumentParser): Main parser object.

		Returns:
			argparse.ArgumentParser: Modified parser object.
		"""
		group = parser.add_argument_group("Model")
		group.add_argument("--projection_head", action="store_true", help="Use projection head")
		group.add_argument("-f", "--freeze", action="store_true", help="Freeze feature extractor weights")

		return parser

	# ----------------------------------------------------------------------------------------------
	def init_layers(self, n_input_channels: int):
		"""
		Initialise layers.

		Args:
			n_input_channels (int): Number of channels in input tensor.
		"""
		# D, H, W
		kernel_size = 3
		stride = (1, 2, 2)
		pad = (0, 1, 1)

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
			conv(64, 64, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
		)

		# Compute shape by doing one forward pass
		self.reset_env()
		with torch.no_grad():
			observation = torch.as_tensor(self.observation_space.sample()[None]).float()
			result = self(observation, calc_n_flatten=True)
			# self.n_flatten = result.shape[1]
			self.n_flatten = math.prod(result.shape)

		if self.projection_head:
			self.layer5 = torch.nn.Sequential(
				torch.nn.Flatten(),
				torch.nn.Linear(self.n_flatten, self.features_dim), # 15360 -> ...
				torch.nn.ReLU(inplace=True),
				# torch.nn.Sigmoid(),
			)
			self.projection = torch.nn.Sequential(
				torch.nn.Linear(self.features_dim, self.projection_dim),
				# torch.nn.CELU(inplace=True),
			)
		else:
			self.layer5 = torch.nn.Sequential(
				torch.nn.Flatten(),
				torch.nn.Linear(self.n_flatten, self.features_dim), # 15360 -> ...
				torch.nn.ReLU(inplace=True),
				# torch.nn.Sigmoid(),
				torch.nn.Linear(self.features_dim, self.projection_dim),
				# torch.nn.CELU(inplace=True),
			)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, calc_n_flatten: bool = False) -> torch.Tensor:
		"""
		Forward pass of model.

		Args:
			x (torch.Tensor): Event stream observation input.
			mask (Optional[torch.Tensor], optional): Mask for partial convolutions. Zero means mask. Defaults to None: mask empty pixels.
			calc_n_flatten (bool, optional): Used to calculate layer sizes during initialisation. Defaults to False.

		Returns:
			torch.Tensor: Extracted features.
		"""
		if mask is None:
			mask = (x != 0).float()

		x, mask = self.process(self.layer1, x, mask, self.layer1_out)
		self.layer1_out = x.detach()
		x, mask = self.process(self.layer2, x, mask, self.layer2_out)
		self.layer2_out = x.detach()
		x, mask = self.process(self.layer3, x, mask, self.layer3_out)
		self.layer3_out = x.detach()
		x, mask = self.process(self.layer4, x, mask, self.layer4_out)
		self.layer4_out = x.detach()

		x = x[:, :, -1]
		if calc_n_flatten:
			return x
		# The sizes of `x` and `mask` will diverge here, but that's OK as we don't need the mask anymore
		# We only care about the final bin prediction for now...
		x = self.layer5(x)

		return x

	# ----------------------------------------------------------------------------------------------
	def project(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Final operation in forward pass, for the projection head. Not performed during normal forward pass.

		Args:
			x (torch.Tensor): Layer input.

		Returns:
			torch.Tensor: Layer output.
		"""
		assert self.projection_head

		return self.projection(x)

	# ----------------------------------------------------------------------------------------------
	def process(self, sequential: torch.nn.Sequential, x: torch.Tensor, mask: Optional[torch.Tensor] = None, previous_output: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
		Conditional operations with support for masks, depending on their layer type.

		Args:
			sequential (torch.nn.Sequential): `torch` `Sequential` object for each layer.
			x (torch.Tensor): Layer input.
			mask (Optional[torch.Tensor], optional): Layer mask of same shape as input. Defaults to None.
			previous_output (Optional[torch.Tensor], optional): Output of previous layer (in time, not depth) for decay convolutions. Defaults to None.

		Returns:
			tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and mask.
		"""
		for module in sequential:
			if isinstance(module, torch.nn.Linear):
				x = x.mean(dim=(3, 4)) # NCDHW -> NCD
				x = x.permute(0, 2, 1) # NCD -> NDC
				x = module(x)
				x = x.permute(0, 2, 1) # NDC -> NCD
			elif isinstance(module, Decay3dPartial):
				if mask is not None:
					x, mask, weights = module(x, mask, previous_output)
			else:
				x = module(x)

		return x, mask

	# ----------------------------------------------------------------------------------------------
	def reset_env(self):
		"""Reset temporal information, to be called when the environment resets."""
		self.layer1_out = None
		self.layer2_out = None
		self.layer3_out = None
		self.layer4_out = None
