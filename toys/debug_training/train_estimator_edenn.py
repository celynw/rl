#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
from typing import Optional, Union
import math
import collections

from setproctitle import setproctitle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from rich import print, inspect

# import rl
from rl.utils.step import Step

# ==================================================================================================
def main():
	setproctitle(Path(__file__).name)
	model = Model(parse_args())
	logger = TensorBoardLogger(
		save_dir=Path(__file__).parent / "runs",
		name=Path(__file__).stem,
		default_hp_metric=False
	)
	trainer = pl.Trainer(
		logger=logger,
		callbacks=[
			callbacks.LearningRateMonitor(),
			callbacks.ModelCheckpoint(save_top_k=3, monitor=f"loss/{Step.VAL}", save_last=True),
		],
	)
	# model.logger.watch(model)
	trainer.fit(model)


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


# ==================================================================================================
class Model(pl.LightningModule):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		super().__init__()
		self.args = args
		self.criterion = torch.nn.L1Loss()
		# self.logger = WandbLogger(project="RL_estimator", log_model="all")

		# D, H, W
		kernel_size = 3
		stride = (1, 2, 2)
		pad = (0, 1, 1)

		partial_kwargs = {}
		conv = Decay3dPartial
		partial_kwargs["multi_channel"] = True
		partial_kwargs["return_mask"] = True
		partial_kwargs["kernel_ratio"] = True

		self.conv1 = torch.nn.Sequential(
			conv(2, 16, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.CELU(inplace=True),
			torch.nn.BatchNorm3d(num_features=16),
		)
		self.conv2 = torch.nn.Sequential(
			conv(16, 32, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.CELU(inplace=True),
			torch.nn.BatchNorm3d(num_features=32),
		)
		self.conv3 = torch.nn.Sequential(
			conv(32, 64, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, return_decay=True, **partial_kwargs),
			torch.nn.CELU(inplace=True),
			torch.nn.BatchNorm3d(num_features=64),
		)
		# self.conv4 = torch.nn.Sequential(
		# 	conv(64, 128, kernel_size=kernel_size, stride=stride, bias=False, padding=pad, return_decay=True, **partial_kwargs),
		# 	torch.nn.CELU(inplace=True),
		# 	torch.nn.BatchNorm3d(num_features=128),
		# )

		# Bias?
		self.mid = torch.nn.Sequential(
			conv(64, 64, kernel_size=(1, 1, 1), stride=1, bias=True, padding=0, return_decay=True, **partial_kwargs),
		)
		self.flatten = torch.nn.Flatten()
		self.final = torch.nn.Sequential(
			torch.nn.Linear(15360, 128),
			torch.nn.CELU(inplace=True),
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
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

		return {
			"optimizer": optimizer,
			"scheduler": scheduler,
			"monitor": f"loss/{Step.VAL}",
		}

	# ----------------------------------------------------------------------------------------------
	def step(self, step: Step, batch, batch_idx: int):
		gt = batch["gt"]
		obs = batch["obs"]

		pred = self(obs)
		loss = self.criterion(pred, gt)
		self.log(f"loss/{step}", loss)

		self.log(f"loss/{step}", loss)
		if batch_idx == 0:
			obs = torch.cat((obs, torch.zeros([obs.shape[0], 1, *obs.shape[2:]])), 1)
			self.logger.experiment.add_image(f"{step}/obs", torchvision.utils.make_grid(obs, nrow=4, pad_value=0.5), global_step=self.current_epoch)
		if step is not Step.TRAIN:
			x_gt, xdot_gt, theta_gt, thetadot_gt = gt[0]
			x_pred, xdot_pred, theta_pred, thetadot_pred = pred[0]
			self.logger.experiment.add_scalar(f"{step}/x_gt", x_gt, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/xdot_gt", xdot_gt, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/theta_gt", theta_gt, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/thetadot_gt", thetadot_gt, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/x_pred", x_pred, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/xdot_pred", xdot_pred, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/theta_pred", theta_pred, global_step=self.current_epoch)
			self.logger.experiment.add_scalar(f"{step}/thetadot_pred", thetadot_pred, global_step=self.current_epoch)

		return loss

	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch, batch_idx: int):
		return self.step(Step.TRAIN, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def validation_step(self, batch, batch_idx: int):
		return self.step(Step.VAL, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def dataloader(self, step: Step):
		return DataLoader(
			EnvDataset(self.args.dataset_dir),
			batch_size=1 if step is Step.TEST else 8,
			shuffle=step is Step.TRAIN,
			drop_last=step is Step.TRAIN,
			pin_memory=True,
		)

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self):
		return self.dataloader(Step.TRAIN)

	# ----------------------------------------------------------------------------------------------
	def val_dataloader(self):
		return self.dataloader(Step.VAL)


# ==================================================================================================
class EnvDataset(Dataset):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, root_dir: Path):
		self.paths = list(root_dir.glob("*.npz"))

	# ----------------------------------------------------------------------------------------------
	def __len__(self):
		return len(self.paths)

	# ----------------------------------------------------------------------------------------------
	def __getitem__(self, idx):
		data = np.load(self.paths[idx], allow_pickle=True)
		obs = data["obs"]
		gt = data["gt"]

		obs = torch.tensor(obs)
		gt = torch.tensor(gt)

		obs = obs.sum(1)

		sample = {
			"gt": gt,
			"obs": obs,
		}

		return sample


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-d", "--dataset_dir", type=Path, default=Path(__file__).parent / "dataset", help="Dataset directory")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main()
