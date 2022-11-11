import math
import collections

import torch
import torch.nn.functional as F

# ==================================================================================================
class Decay3d(torch.nn.modules.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] = 3,
			stride: int | tuple[int, int, int] = 1, padding: int | tuple[int, int, int] = 0,
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
