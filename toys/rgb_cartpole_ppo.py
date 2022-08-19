#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import math
import time
import collections
from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
from typing import Union, Optional

import torch
import torch.nn.functional as F
import gym
from stable_baselines3.common.monitor import Monitor
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import wandb
from wandb.integration.sb3 import WandbCallback
from tqdm import tqdm
from rich import print, inspect

import rl
# from rl.models.utils.visual import CartPoleRGBTemp
from rl.models import PPO_mod, A2C_mod, Estimator
from rl.models.utils import ActorCriticPolicy_mod
from rl.utils import TqdmCallback, PolicyUpdateCallback

use_wandb = False

# ==================================================================================================
class Decay3d(torch.nn.modules.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple[int, int]] = 3,
			stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0,
			bias: bool = True, spatial: tuple[int, int] = (1, 1)):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (1, kernel_size, kernel_size)
		self.stride = stride if isinstance(stride, collections.abc.Iterable) else (1, stride, stride)
		self.padding = padding if isinstance(padding, collections.abc.Iterable) else (0, padding, padding)
		assert self.kernel_size[0] == 1
		assert self.stride[0] == 1
		if bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter("bias", None)

		self.conv_weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
		self.decay_weight = torch.nn.Parameter(torch.Tensor(out_channels, *spatial))

		self.reset_parameters()

	# ----------------------------------------------------------------------------------------------
	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
		torch.nn.init.kaiming_uniform_(self.decay_weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias, -bound, bound)

	# ----------------------------------------------------------------------------------------------
	def extra_repr(self):
		return f"in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None}"

	# ----------------------------------------------------------------------------------------------
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		assert len(input.shape) == 5 # NCDHW
		# First, propagate values from previous layer using 2D convolution
		# Use first part of weights
		output = F.conv3d(input, self.conv_weight, bias=None, stride=self.stride, padding=self.padding)

		# Now, propagate decay values from resulting tensor
		output = output.permute(2, 0, 1, 3, 4) # NCDHW -> DNCHW
		# Combine signals from previous layers, and residual signals from current layer but earlier in time
		# Only want positive values for the decay factor
		decay = self.decay_weight / (1 + abs(self.decay_weight))
		for i, d in enumerate(output):
			if i == 0:
				continue
			output[i] = output[i].clone() + output[i - 1].clone() * decay
		output = output.permute(1, 2, 0, 3, 4) # DNCHW -> NCDHW
		if self.bias is not None:
			output = output + self.bias.view(1, self.out_channels, 1, 1, 1)

		return output


# ==================================================================================================
class Decay3dPartial(Decay3d):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple[int, int]] = 3,
			stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0,
			bias: bool = True, multi_channel: bool = False, return_mask: bool = True,
			return_decay: bool = False, kernel_ratio: bool = False, spatial: tuple[int, int] = (1, 1),
			scale_factor: Union[int, tuple[int, int, int]] = 1):
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, spatial)
		self.multi_channel = multi_channel
		self.return_mask = return_mask
		self.return_decay = return_decay
		self.kernel_ratio = kernel_ratio
		self.scale_factor = scale_factor
		if self.multi_channel:
			self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
		else:
			self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
		self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3] * self.weight_maskUpdater.shape[4]
		self.last_size = (None, None, None, None, None)
		self.update_mask = None
		self.mask_ratio = None

	# ----------------------------------------------------------------------------------------------
	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
		torch.nn.init.kaiming_uniform_(self.decay_weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias, -bound, bound)

	# ----------------------------------------------------------------------------------------------
	def forward(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None, previous_output: Optional[torch.Tensor] = None) -> torch.Tensor:
		# First, propagate values from previous layer using 2D convolution
		# Use first part of weights
		# Partial convolutions
		assert len(input.shape) == 5
		if self.scale_factor != 1:
			mask_in = torch.nn.Upsample(scale_factor=self.scale_factor, mode="nearest")(mask_in)
			assert self.scale_factor == (1, 2, 2) # TODO implement for other scale factors..!
			# mask_in = torch.nn.Upsample(scale_factor=self.scale_factor, mode="nearest")(mask_in)[:, :, 1:-1, 1:-1, :]
		if mask_in is not None:
			try:
				assert input.shape == mask_in.shape
			except AssertionError:
				error(f"Input/mask mismatch in partial convolution: {input.shape} vs. {mask_in.shape}")
				raise

		if mask_in is not None or self.last_size != tuple(input.shape):
			self.last_size = tuple(input.shape)

			with torch.no_grad():
				if self.weight_maskUpdater.type() != input.type():
					self.weight_maskUpdater = self.weight_maskUpdater.to(input)

				if mask_in is None:
					# if mask is not provided, create a mask
					if self.multi_channel:
						mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3], input.data.shape[4]).to(input)
					else:
						mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3], input.data.shape[4]).to(input)
				else:
					mask = mask_in

				self.update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, groups=1)

				if self.kernel_ratio:
					# Normal behaviour:
					# 1. Multiply by self.slide_winsize (the maximum number of cells I could see)
					# 2. Divide by self.update_mask (the number of cells I can see; not masked)
					# Instead:
					# 1. Multiply by sum of kernel
					# 2. Then divide by sum of unmasked kernel
					assert(len(self.stride) == 3 and self.stride[0] == 1)
					assert(len(self.padding) == 3 and self.padding[0] == 0)
					ratio_masked = F.conv3d(torch.ones_like(input) * mask, self.conv_weight.data, bias=None, stride=self.stride, padding=self.padding, groups=1)
					# ones_like() may be redundant, but let's be sure
					ratio_unmasked = F.conv3d(torch.ones_like(input), self.conv_weight.data, bias=None, stride=self.stride, padding=self.padding, groups=1)

					# For mixed precision training, change 1e-8 to 1e-6
					self.mask_ratio = ratio_masked / (ratio_unmasked + 1e-8)
					self.update_mask = torch.clamp(self.update_mask, 0, 1)
					self.mask_ratio = self.mask_ratio * self.update_mask
				else:
					# For mixed precision training, change 1e-8 to 1e-6
					self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
					self.update_mask = torch.clamp(self.update_mask, 0, 1)
					self.mask_ratio = self.mask_ratio * self.update_mask

		raw_out = F.conv3d(input * mask_in, self.conv_weight, self.bias, self.stride, self.padding)

		if self.bias is not None:
			bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
			output = ((raw_out - bias_view) * self.mask_ratio) + bias_view
			output = output * self.update_mask
		else:
			output = raw_out * self.mask_ratio

		# Now, propagate decay values from resulting tensor
		output = output.permute(2, 0, 1, 3, 4) # NCDHW -> DNCHW
		# Combine signals from previous layers, and residual signals from current layer but earlier in time
		# Only want positive values for the decay factor
		decay = self.decay_weight / (1 + abs(self.decay_weight))
		for i, d in enumerate(output):
			if i == 0:
				if previous_output is not None:
					# TODO move this somewhere else or clean up somehow!
					previous_output = previous_output.permute(2, 0, 1, 3, 4) # NCDHW -> DNCHW

					# FIX Not sure why sometimes during policy update, N == 1 rather than full (64)
					# Dodgy hack
					if previous_output.shape[1] != output.shape[1]:
						previous_output = previous_output[:, -output.shape[1]:]
					try:
						output[i] = previous_output[-1].clone() * decay
					except RuntimeError:
						print(f"output: {output.shape}, previous_output: {previous_output.shape}, decay: {decay.shape}")
						raise
				else:
					continue
			output[i] = output[i].clone() + output[i - 1].clone() * decay
		output = output.permute(1, 2, 0, 3, 4) # DNCHW -> NCDHW
		if self.bias is not None:
			output = output + self.bias.view(1, self.out_channels, 1, 1, 1)

		returns = [output]
		if self.return_mask:
			returns.append(self.update_mask)
		if self.return_decay: # FIX I don't think I need to do this! self.decay weight should be the same?
			# returns.append(decay)
			returns.append(self.decay_weight.detach())
		if len(returns) == 1:
			returns = returns[0]

		return returns


# # ==================================================================================================
# class InfoCallback(BaseCallback):
# 	# ----------------------------------------------------------------------------------------------
# 	def __init__(self):
# 		super().__init__()

# 	# ----------------------------------------------------------------------------------------------
# 	def _on_training_start(self):
# 		self.fails = {
# 			"too_far_left": 0,
# 			"too_far_right": 0,
# 			"pole_fell_left": 0,
# 			"pole_fell_right": 0,
# 		}

# 	# ----------------------------------------------------------------------------------------------
# 	def _on_step(self):
# 		info = self.locals["info"]
# 		if info["failReason"] is not None:
# 			self.fails[info["failReason"]] += 1

# 		return True


# # ==================================================================================================
# class EDeNN(torch.nn.Module):
# 	"""
# 	Custom network for policy and value function.
# 	It receives as input the features extracted by the feature extractor.

# 	:param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
# 	:param last_layer_dim_pi: (int) number of units for the last layer of the policy network
# 	:param last_layer_dim_vf: (int) number of units for the last layer of the value network
# 	"""
# 	# ----------------------------------------------------------------------------------------------
# 	def __init__(
# 		self,
# 		feature_dim: int,
# 		last_layer_dim_pi: int = 64,
# 		last_layer_dim_vf: int = 64,
# 	):
# 		super().__init__()

# 		# IMPORTANT:
# 		# Save output dimensions, used to create the distributions
# 		self.latent_dim_pi = last_layer_dim_pi
# 		self.latent_dim_vf = last_layer_dim_vf

# 		# Policy network
# 		self.policy_net = self.get_layers(feature_dim, last_layer_dim_pi)

# 		# Value network
# 		self.value_net = self.get_layers(feature_dim, last_layer_dim_vf)

# 	# ----------------------------------------------------------------------------------------------
# 	def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
# 		"""
# 		:return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
# 			If all layers are shared, then ``latent_policy == latent_value``
# 		"""
# 		return self.policy_net(features), self.value_net(features)

# 	# ----------------------------------------------------------------------------------------------
# 	def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
# 		return self.policy_net(features)

# 	# ----------------------------------------------------------------------------------------------
# 	def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
# 		return self.value_net(features)

# 	# ----------------------------------------------------------------------------------------------
# 	def get_layers(self, feature_dim: int, last_layer_dim: int):
# 		# D, H, W
# 		kernel_size = 3
# 		stride = (1, 2, 2)
# 		pad = (0, 1, 1)
# 		scale_factor = (1, 2, 2)

# 		partial_kwargs = {}
# 		conv = Decay3dPartial
# 		partial_kwargs["multi_channel"] = True
# 		partial_kwargs["return_mask"] = True
# 		partial_kwargs["kernel_ratio"] = True

# 		self.conv1 = torch.nn.Sequential(
# 			conv(feature_dim, 16, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, **partial_kwargs),
# 			torch.nn.CELU(inplace=True),
# 			torch.nn.BatchNorm3d(num_features=16),
# 		)
# 		self.conv2 = torch.nn.Sequential(
# 			conv(16, 32, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, **partial_kwargs),
# 			torch.nn.CELU(inplace=True),
# 			torch.nn.BatchNorm3d(num_features=32),
# 		)
# 		self.conv3 = torch.nn.Sequential(
# 			conv(32, 64, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, **partial_kwargs),
# 			torch.nn.CELU(inplace=True),
# 			torch.nn.BatchNorm3d(num_features=64),
# 		)
# 		self.conv4 = torch.nn.Sequential(
# 			conv(64, 128, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, **partial_kwargs),
# 			torch.nn.CELU(inplace=True),
# 			torch.nn.BatchNorm3d(num_features=128),
# 		)

# 		self.mid = torch.nn.Sequential(
# 			conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=False, padding=0, **partial_kwargs),
# 			conv(128, 128, kernel_size=(1, 1, 1), stride=1, bias=False, padding=0, **partial_kwargs),
# 		)

# 		self.final = torch.nn.Linear(128, last_layer_dim, bias=True)

# 		return torch.nn.Sequential(
# 			self.conv1,
# 			self.conv2,
# 			self.conv3,
# 			self.conv4,
# 			self.mid,
# 		)


# # ==================================================================================================
# class EDeNNPolicy(ActorCriticPolicy):
# 	def __init__(
# 		self,
# 		observation_space: gym.spaces.Space,
# 		action_space: gym.spaces.Space,
# 		lr_schedule: Callable[[float], float],
# 		net_arch: Optional[list[Union[int, dict[str, list[int]]]]] = None,
# 		# activation_fn: torch.nn.Module = torch.nn.Tanh,
# 		activation_fn: torch.nn.Module = torch.nn.CELU,
# 		*args,
# 		**kwargs,
# 	):
# 		super().__init__(
# 			observation_space,
# 			action_space,
# 			lr_schedule,
# 			net_arch,
# 			activation_fn,
# 			# Pass remaining arguments to base class
# 			*args,
# 			**kwargs,
# 		)
# 		# Disable orthogonal initialization
# 		self.ortho_init = False

# 	def _build_mlp_extractor(self) -> None:
# 		self.mlp_extractor = EDeNN(self.features_dim)


# ==================================================================================================
class EDeNNCNN(BaseFeaturesExtractor):
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


# # ==================================================================================================
# class Monitor_reset(Monitor):
# 	# ----------------------------------------------------------------------------------------------
# 	def __init__(self, env: gym.Env, filename: Optional[str] = None, allow_early_resets: bool = True, reset_keywords: tuple[str, ...] = (), info_keywords: tuple[str, ...] = ()):
# 		super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)


# # ==================================================================================================
# class FailMonitor(Monitor):
# 	# ----------------`------------------------------------------------------------------------------
# 	def __init__(self, *args, **kwargs):
# 		super.__init__(*args, **kwargs)
#         if filename is not None:
#             self.results_writer = ResultsWriter(
#                 filename,
#                 header={"t_start": self.t_start, "env_id": env.spec and env.spec.id},
#                 extra_keys=reset_keywords + info_keywords,
#             )
#         else:
#             self.results_writer = None
# 		self.fails = {
# 			"too_far_left": 0,
# 			"too_far_right": 0,
# 			"pole_fell_left": 0,
# 			"pole_fell_right": 0,
# 		}

# 	# ----------------------------------------------------------------------------------------------
# 	def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
# 		"""
# 		Step the environment with the given action

# 		:param action: the action
# 		:return: observation, reward, done, information
# 		"""
# 		observation, reward, done, info = super().step(action)
# 		if done:
# 			info = self.locals["info"]
# 			if info["failReason"]:
# 				self.fails[info["failReason"]] += 1
# 			ep_rew = sum(self.rewards)
# 			ep_len = len(self.rewards)
# 			ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
# 			for key in self.info_keywords:
# 				ep_info[key] = info[key]
# 			self.episode_returns.append(ep_rew)
# 			self.episode_lengths.append(ep_len)
# 			self.episode_times.append(time.time() - self.t_start)
# 			ep_info.update(self.current_reset_info)
# 			if self.results_writer:
# 				self.results_writer.write_row(ep_info)
# 			info["episode"] = ep_info
# 		self.total_steps += 1
# 		return observation, reward, done, info

# 	def close(self) -> None:
# 		"""
# 		Closes the environment
# 		"""
# 		super(Monitor, self).close()
# 		if self.results_writer is not None:
# 			self.results_writer.close()

# 	def get_total_steps(self) -> int:
# 		"""
# 		Returns the total number of timesteps

# 		:return:
# 		"""
# 		return self.total_steps

# 	def get_episode_rewards(self) -> List[float]:
# 		"""
# 		Returns the rewards of all the episodes

# 		:return:
# 		"""
# 		return self.episode_returns

# 	def get_episode_lengths(self) -> List[int]:
# 		"""
# 		Returns the number of timesteps of all the episodes

# 		:return:
# 		"""
# 		return self.episode_lengths

# 	def get_episode_times(self) -> List[float]:
# 		"""
# 		Returns the runtime in seconds of all the episodes

# 		:return:
# 		"""
# 		return self.episode_times


# ==================================================================================================
def main(args: argparse.Namespace) -> None:
	# env_id = "CartPole-contrast-v1"
	env_id = "CartPole-events-v1"
	# env_id = "CartPole-events-debug"
	# env_id = "CartPole-v1"

	name = f"{int(time.time())}" # Epoch
	if args.name:
		name = f"{name} {args.name}"
	args.log_dir /= name
	print(f"Logging to {args.log_dir}")
	args.log_dir.mkdir(parents=True, exist_ok=True)

	config = {
		"policy_type": ActorCriticPolicy_mod,
		"total_timesteps": max(args.n_steps, args.steps),
		"env_name": env_id,
	}
	if use_wandb:
		run = wandb.init(
			project="RL_pretrained",
			config=config,
			sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
			monitor_gym=True, # auto-upload the videos of agents playing the game
			save_code=True, # optional
		)

	all_rewards = []
	# env_ = gym.make(env_id)
	# env_ = CartPoleRGBTemp(env_)
	# env_ = DummyVecEnv([lambda: gym.make(env_id)])

	env_ = gym.make(env_id)

	# def make_env():
	# 	return Monitor(env_, str(args.log_dir), allow_early_resets=True, info_keywords=("failReason", "updatedPolicy"))
	# env = DummyVecEnv([make_env])
	# env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

	# while True:
	# 	obs = env_.reset()
	# print(obs)
	# obs = np.transpose(obs, (1, 2, 0))
	# import cv2
	# cv2.imwrite(f"obs.png", obs)
	# quit(0)

	for i in tqdm(range(args.times)):
		env = Monitor(env_, str(args.log_dir), allow_early_resets=True, info_keywords=("failReason", "updatedPolicy"))
		# env = Monitor_reset(env_, str(args.log_dir), allow_early_resets=True)
		# model = A2C(MlpPolicy, env, verbose=1) # total_timesteps will be at least n_steps (2048)
		# model = PPO(MlpPolicy, env, verbose=1) # total_timesteps will be at least n_steps (2048)
		# model = PPO(EDeNNPolicy, env, verbose=1) # total_timesteps will be at least n_steps (2048)
		policy_kwargs = dict(
			# features_extractor_class=EDeNNCNN,
			features_extractor_class=Estimator,
			# # # features_extractor_kwargs=dict(features_dim=128),
			# # # features_extractor_kwargs=dict(features_dim=18),
			# # features_extractor_kwargs=dict(features_dim=64),
			# features_extractor_kwargs=dict(features_dim=4),
			# net_arch: [{'pi': [64, 64], 'vf': [64, 64]}] # DEFAULT
			# optimizer_class=torch.optim.Adam,
			optimizer_class=torch.optim.SGD,
		)
		model = PPO_mod(
		# model = A2C_mod(
			ActorCriticPolicy_mod,
			env,
			policy_kwargs=policy_kwargs,
			verbose=1,
			learning_rate=args.lr,
			device="cpu" if args.cpu else "auto",
			n_steps=args.n_steps,
			tensorboard_log=args.log_dir,
			# pl_coef=0.0,
			# ent_coef=0.0,
			# vf_coef=0.0,
			bs_coef=0.0,
		) # total_timesteps will be at least n_steps (2048)
		# model = PPO(MlpPolicy, env, verbose=1, learning_rate=args.lr) # total_timesteps will be at least n_steps (2048)
		# inspect(model.policy.features_extractor, all=True)
		# checkpoint = torch.load("/home/cw0071/dev/python/rl/toys/debug_training/runs/train_estimator/version_10/checkpoints/epoch=58-step=7375.ckpt")
		# checkpoint = torch.load("/code/toys/debug_training/runs/train_estimator/version_10/checkpoints/epoch=58-step=7375.ckpt")
		# checkpoint = torch.load("/home/cw0071/dev/python/rl/train_estimator_RL_estimator/219_1e4b47kx/checkpoints/epoch=59-step=75000.ckpt")

		checkpoint = torch.load("/code/train_estimator_RL_estimator/219_1e4b47kx/checkpoints/epoch=59-step=75000.ckpt") # Normal sum(1)
		# checkpoint = torch.load("/code/train_estimator_RL_estimator/221_k1db1ttd/checkpoints/epoch=36-step=46250.ckpt") # Binary->255 image
		# model.policy.features_extractor.load_state_dict(checkpoint["state_dict"], strict=False) # Ignore final layer
		model.policy.features_extractor.load_state_dict(checkpoint["state_dict"])

		# TRY LOADING OPTIMIZER STATE DICT
		# load_state_dict(model.policy.optimizer, checkpoint["optimizer_states"][0])

		# Critic network
		# Could to `model.load` with `custom_objects` parameter, but this isn't done in-place!
		# So...
		import zipfile
		import io
		archive = zipfile.ZipFile("/runs/1657548723 vanilla longer/rgb.zip", "r")
		bytes = archive.read("policy.pth")
		bytes_io = io.BytesIO(bytes)
		stateDict = torch.load(bytes_io)
		# Could load them manually, but it's not that easy
		# model.policy. .load_state_dict(stateDict)
		# 	mlp_extractor.policy_net
		# 	mlp_extractor.value_net
		# 	action_net
		# 	value_net
		# Can't load zip manually with `model.set_parameters`, but can specify a dict
		# Hide some weights we don't want to load
		# Everything under `mlp_extractor`, `action_net` and `policy_net` are in both vanilla and mine!
		# inspect(stateDict, all=True)
		# del stateDict["mlp_extractor.policy_net.0.weight"]
		# del stateDict["mlp_extractor.policy_net.0.bias"]
		# del stateDict["mlp_extractor.policy_net.2.weight"]
		# del stateDict["mlp_extractor.policy_net.2.bias"]
		# del stateDict["mlp_extractor.value_net.0.weight"]
		# del stateDict["mlp_extractor.value_net.0.bias"]
		# del stateDict["mlp_extractor.value_net.2.weight"]
		# del stateDict["mlp_extractor.value_net.2.bias"]
		# del stateDict["action_net.weight"]
		# del stateDict["action_net.bias"]
		# del stateDict["value_net.weight"]
		# del stateDict["value_net.bias"]
		model.set_parameters({"policy": stateDict}, exact_match=False) # FIX not doing "policy.optimizer" key

		# Freeze weights?
		if args.freeze:
			for param in model.policy.features_extractor.parameters():
				param.requires_grad = False
			# model.policy.features_extractor.layer1.0.weight.requires_grad = False
			# model.policy.features_extractor.layer1.0.bias.requires_grad = False
			# FIX can't work out how to do this
			# optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

		env.set_model(model.policy.features_extractor)

		# DEBUG observation space
		# env_.reset()
		# obs = env_.step(1)
		# while obs[2] is False:
		# 	obs = env_.step(1)
		# print(np.unique(obs[0]))
		# print(np.unique(obs[0], return_counts=True))
		# quit(0)




		# visualisation = {}

		# def hook_fn(module, input, output):
		# 	visualisation[module] = output

		# def get_all_layers(net):
		# 	for name, layer in net._modules.items():
		# 		#If it is a sequential, don't register a hook on it
		# 		# but recursively register hook on all it's module children
		# 		if isinstance(layer, torch.nn.Sequential):
		# 			get_all_layers(layer)
		# 		else:
		# 			layer.register_forward_hook(hook_fn)

		# get_all_layers(model.policy)
		# model.policy(torch.randn([1, 2, 64, 240], device="cuda"))
		# # print(visualisation.keys())
		# print(visualisation)
		# quit(0)



		# model = PPO(
		# 	MlpPolicy,
		# 	env,
		# 	verbose=1,
		# 	# Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
		# 	# Equivalent to classic advantage when set to 1.
		# 	# https://arxiv.org/abs/1506.02438
		# 	# To obtain Monte-Carlo advantage estimate
		# 	# 	(A(s) = R - V(S))
		# 	# where R is the sum of discounted reward with value bootstrap
		# 	# (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization
		# 	# gae_lambda=1.0,
		# 	# clip_range=0.0,
		# 	# max_grad_norm=1.0,
		# )
		if use_wandb:
			wandbCallback = WandbCallback(
				model_save_path=args.log_dir / f"{args.name}",
				gradient_save_freq=100,
			)
			callback = [TqdmCallback(), wandbCallback, PolicyUpdateCallback(env)]
		else:
			callback = [TqdmCallback(), PolicyUpdateCallback(env)]
		model.learn(
			total_timesteps=max(model.n_steps, args.steps),
			# callback=[TqdmCallback()],
			callback=callback,
			eval_freq=args.n_steps,
		)
		all_rewards.append(env.get_episode_rewards())

		if args.save and i == 0:
			# model.save(args.log_dir / "a2c_cartpole_rgb")
			model.save(args.log_dir / f"{args.name}")

		if args.render and i == args.times - 1:
			obs = env.reset()
			for i in range(1000):
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)
				env.render()
				if done:
					obs = env.reset()
	env.close()
	# run.finish()

	# all_rewards_ = -np.ones([len(all_rewards), max([len(x) for x in all_rewards])])
	# for i, sub in enumerate(all_rewards):
	# 	all_rewards_[i][0:len(sub)] = sub
	# all_rewards = np.ma.MaskedArray(all_rewards_, mask=all_rewards_ < 0)

	# mean = all_rewards.mean(axis=0)
	# std = all_rewards.std(axis=0)
	# ax = plt.gca()
	# ax.set_ylim([0, 500]) # TODO get env max reward instead of hard coding
	# ax.fill_between(
	# 	range(all_rewards.shape[1]), # X
	# 	# np.clip(mean - std, 0, None), # Max
	# 	mean - std, # Max
	# 	mean + std, # Min
	# 	alpha=.5,
	# 	linewidth=0
	# )
	# ax.plot(mean, linewidth=2)
	# plt.savefig(str(args.log_dir / f"plot.png"))


# ==================================================================================================
# Copied from torch/optim/optimizer.py version 1.11.0
# Suppressing the ValueError!
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
		print("### SUPPRESSED ValueError ###")
		print("loaded state dict contains a parameter group "
							"that doesn't match the size of optimizer's group")
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
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("times", type=int, help="Times to run")
	parser.add_argument("-r", "--render", action="store_true", help="Render final trained model output")
	parser.add_argument("-s", "--steps", type=int, default=1000, help="How many steps to train for")
	parser.add_argument("-S", "--save", action="store_true", help="Save first trained model")
	parser.add_argument("-n", "--name", type=str, help="Name of experiment")
	parser.add_argument("-d", "--log_dir", type=Path, default=Path("/tmp/gym/"), help="Location of log directory")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
	parser.add_argument("-f", "--freeze", action="store_true", help="Freeze feature extractor weights")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps before each weights update")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
