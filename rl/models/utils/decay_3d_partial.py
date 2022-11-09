import math
from typing import Optional

import torch
import torch.nn.functional as F
from rich import print, inspect

from rl.models.utils import Decay3d

# ==================================================================================================
class Decay3dPartial(Decay3d):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] = 3,
			stride: int | tuple[int, int, int] = 1, padding: int | tuple[int, int, int] = 0,
			bias: bool = True, multi_channel: bool = False, return_mask: bool = True,
			return_decay: bool = False, kernel_ratio: bool = False, spatial: tuple[int, int] = (1, 1),
			scale_factor: int | tuple[int, int, int] = 1):
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
