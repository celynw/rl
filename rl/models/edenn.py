#!/usr/bin/env python3
import argparse
import math
from typing import Optional

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rich import print, inspect

from rl.models.utils import Decay3dPartial

# ==================================================================================================
class EDeNN(BaseFeaturesExtractor):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: spaces.Box, features_dim: int, projection_head: Optional[int] = None):
		"""
		Feature extractor using "EDeNN: Event Decay Neural Networks for low latency vision" (Celyn Walters, Simon Hadfield).
		https://arxiv.org/abs/2209.04362

		Args:
			observation_space (spaces.Box): Observation space from environment.
			features_dim (int): Output size of final layer (features) for RL policy.
			projection_head (int, optional): Output size of projection head layer, or disable. Defaults to None.
		"""
		self.observation_space = observation_space
		self.num_bins = self.observation_space.shape[1]
		super().__init__(observation_space, features_dim)
		self.projection_head = projection_head
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
		group.add_argument("--projection_head", type=int, default=None, help="Use projection head, number is size of layer before projection which is sent to policy network")
		# group.add_argument("-f", "--freeze", action="store_true", help="Freeze feature extractor weights")

		return parser

	# ----------------------------------------------------------------------------------------------
	def init_layers(self, n_input_channels: int):
		"""
		Initialise layers.

		Args:
			n_input_channels (int): Number of channels in input tensor.
		"""
		partial_kwargs = {}
		if isinstance(conv, Decay3dPartial):
			partial_kwargs["multi_channel"] = True
			partial_kwargs["return_mask"] = True
			partial_kwargs["kernel_ratio"] = True

		self.layer1 = torch.nn.Sequential(
			conv(n_input_channels, 32, kernel_size=8, stride=4, bias=True, padding=0, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			# torch.nn.BatchNorm3d(num_features=32),
		)
		self.layer2 = torch.nn.Sequential(
			conv(32, 64, kernel_size=4, stride=2, bias=True, padding=0, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			# torch.nn.BatchNorm3d(num_features=64),
		)
		self.layer3 = torch.nn.Sequential(
			conv(64, 64, kernel_size=3, stride=1, bias=True, padding=0, **partial_kwargs),
			torch.nn.ReLU(inplace=True),
			# torch.nn.BatchNorm3d(num_features=64),
		)

		# Compute shape by doing one forward pass
		with torch.no_grad():
			observation = torch.as_tensor(self.observation_space.sample()[None]).float()
			result = self(observation, [torch.tensor([0]), torch.tensor([0]), torch.tensor([0])], calc_n_flatten=True)
			# self.n_flatten = result.shape[1]
			self.n_flatten = math.prod(result.shape)

		self.layer_last = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(self.n_flatten, self.features_dim),
			torch.nn.ReLU(inplace=True),
			# torch.nn.Sigmoid(),
			# torch.nn.CELU(inplace=True),
		)

		if self.projection_head is not None:
			self.projection = torch.nn.Sequential(
				torch.nn.Linear(self.features_dim, self.projection_head),
				# torch.nn.CELU(inplace=True),
			)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor, prev_x: list[torch.Tensor], calc_n_flatten: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
		"""
		Forward pass of model.

		Args:
			x (torch.Tensor): Event stream observation input.
			mask (Optional[torch.Tensor], optional): Mask for partial convolutions. Zero means mask. Defaults to None: mask empty pixels.
			calc_n_flatten (bool, optional): Used to calculate layer sizes during initialisation. Defaults to False.

		Returns:
			torch.Tensor: Extracted features.
		"""
		new_prev_x = []
		mask = (x != 0).float()

		x, mask = self.process(self.layer1, x, mask, prev_x[0])
		if not calc_n_flatten:
			new_prev_x.append(x.detach()[:, :, -1])
		x, mask = self.process(self.layer2, x, mask, prev_x[1])
		if not calc_n_flatten:
			new_prev_x.append(x.detach()[:, :, -1])
		x, mask = self.process(self.layer3, x, mask, prev_x[2])
		if not calc_n_flatten:
			new_prev_x.append(x.detach()[:, :, -1])
		# x, mask = self.process(self.layer4, x, mask, prev_x[3])
		# if not calc_n_flatten:
		# 	new_prev_x.append(x.detach())

		x = x[:, :, -1:] # Only consider the last time bin, but keep dimension
		# x = x[:, :, self.num_bins - 1::self.num_bins] # Only consider the last time bin FOR EACH WINDOW
		if calc_n_flatten:
			return x
		# Similar to `RolloutBuffer.swap_and_flatten()`, but we already did some steps
		# Result needs to be shaped like (all time steps) * environments
		x = x.permute(0, 2, 1, 3, 4)
		x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

		# The sizes of `x` and `mask` will diverge here, but that's OK as we don't need the mask anymore
		x = self.layer_last(x)

		# x: [1, self.features_dim]
		return x, new_prev_x

	# ----------------------------------------------------------------------------------------------
	def project(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Final operation in forward pass, for the projection head. Not performed during normal forward pass.

		Args:
			x (torch.Tensor): Layer input.

		Returns:
			torch.Tensor: Layer output.
		"""
		assert self.projection_head is not None

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


# ==================================================================================================
if __name__ == "__main__":
	import gymnasium as gym
	from rich import print, inspect

	import rl
	import rl.environments

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser = EDeNN.add_argparse_args(parser)
	parser = rl.environments.CartPoleEvents.add_argparse_args(parser)
	args = parser.parse_args()
	args.projection_head = 256

	env = gym.make(
		"CartPoleEvents-v0",
		args=args,
	)
	edenn = EDeNN(observation_space=env.observation_space, features_dim=env.state_space.shape[-1], projection_head=args.projection_head)

	event_tensor = torch.rand([1, 2, args.tsamples, env.output_height, env.output_width])
	print(f"input event shape: {event_tensor.shape}")
	print(f"features_dim: {edenn.features_dim}")

	edenn = edenn.to("cuda")
	event_tensor = event_tensor.to("cuda")
	output = edenn(event_tensor)

	print(f"model: {edenn}")
	print(f"output features shape: {output.shape}")
