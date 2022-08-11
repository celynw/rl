import argparse
import os
import platform
import inspect
from typing import Optional, Any

import torch
import pytorch_lightning as pl
import gym
from kellog import info, warning, error, debug
import numpy as np
import optuna

import rl
from rl import utils
from rl import metrics
from rl.utils import Step, Dir
from .utils import ModelType, choose_model
from .utils.visual import MaxAndSkipEnv, FireResetEnv, WarpFrame, ImageToPyTorch, FrameStack, ScaledFloatFrame

# ==================================================================================================
class Base(pl.LightningModule):
	monitor = f"loss/{Step.VAL}"
	monitor_dir = Dir.MIN
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None):
		super().__init__()
		self.save_hyperparameters(args)
		self.trial = trial
		self.metric_inference_time_train = metrics.InferenceTime()
		self.metric_inference_time_val = metrics.InferenceTime()
		self.metric_inference_time_test = metrics.InferenceTime()
		np.set_printoptions(precision=3, linewidth=os.get_terminal_size().columns)

		self.model_type = choose_model(self.hparams.env)

	# ----------------------------------------------------------------------------------------------
	def create_env(self) -> gym.Env:
		env = gym.make(self.hparams.env)
		if self.model_type is ModelType.VISUAL:
			fire = True
			env = MaxAndSkipEnv(env) # Return only every `skip`-th frame
			if fire:
				env = FireResetEnv(env) # Fire at the beginning
			env = WarpFrame(env) # Reshape image
			env = ImageToPyTorch(env) # Invert shape
			env = FrameStack(env, 4) # Stack last 4 frames
			env = ScaledFloatFrame(env) # Scale frames

		return env

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Add model-specific arguments.

		Args:
			parser (argparse.ArgumentParser): Main parser

		Returns:
			argparse.ArgumentParser: Modified parser
		"""
		group = parser.add_argument_group("Model")
		# TODO get list of valid datasets from model definition itself
		datasets = [name for name, obj in inspect.getmembers(rl.datasets) if inspect.isclass(obj) and name != "Base"]
		group.add_argument("dataset", choices=datasets, metavar=f"DATASET: {{{', '.join(datasets)}}}", help="Dataset to train/val/test on")
		args_known, _ = parser.parse_known_args()
		if args_known.dataset is None:
			parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

		# group.add_argument()

		Dataset = getattr(rl.datasets, args_known.dataset)
		parser = Dataset.add_argparse_args(parser)

		return parser

	# ----------------------------------------------------------------------------------------------
	def on_train_start(self):
		if self.logger:
			tensorboard = self.logger.experiment
			tensorboard.add_text("Git revision", utils.get_git_rev())

			# Remember that you need to add two spaces before line breaks in `add_text()`...
			gpus = utils.get_gpu_info()
			string = f"hostname: {platform.uname()[1]}"
			string += f"  \ngpus:"
			if gpus:
				for gpu in gpus:
					string += f"  \n- {gpu['name']} ({gpu['memory']}GB) [{gpu['capability']}]"
			else:
				string += " None"
			tensorboard.add_text("System info", string)

	# # ----------------------------------------------------------------------------------------------
	# def on_train_batch_start(self, batch: Any, batch_idx: int):
	# 	# For attaching logged loss to hparams in tensorboard
	# 	# Make sure first entry is not zero
	# 	# When this is called, it might also affect the real logged losses, so makes sure to not overwrite those
	# 	# Only the first call works, so we need as many loss types as we can before we call this
	# 	# hparams tab in tensorboard is a separate
	# 	if self.logger and self.current_epoch == 1:
	# 		hyperparams = {}
	# 		for hparam in [f"loss/{Step.TRAIN}", f"loss/{Step.VAL}", f"loss/{Step.TEST}"]:
	# 			if hparam in self.trainer.callback_metrics:
	# 				hyperparams[hparam] = self.trainer.callback_metrics[hparam]
	# 		# self.logger.log_hyperparams(self.args, hyperparams)
	# 		# self.logger.log_hyperparams()

	# ----------------------------------------------------------------------------------------------
	def is_training(self):
		# Generating tensorboard graph with example_input_array: model.training
		# Sanity check: self.trainer.sanity_checking()
		# Normal training: model.training and model.trainer.training
		# Normal validation: self.trainer.validating
		# Normal testing: self.trainer.testing? TODO verify
		if self.trainer is None:
			return False
		return (self.training and self.trainer.training) or self.trainer.validating or self.trainer.testing

	# ----------------------------------------------------------------------------------------------
	def configure_optimizers(self):
		"""
		Set up optimisers.

		Returns:
			Tuple[list[torch.optim.Optimizer], list[object]]: Optimiser(s) and learning rate scheduler(s)
		"""
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, amsgrad=True)
		lr_scheduler = {
			"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True),
			"monitor": self.monitor,
		}

		return [optimizer], [lr_scheduler]

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
		"""Forward pass."""
		for layer in self.layers:
			# Sometimes (during summarisation) x and mask get combined anyway. Decouple them here
			while isinstance(x, tuple):
				assert len(x) == 2
				x, mask = x
			if isinstance(layer, torch.nn.Sequential):
				for sublayer in layer:
					x, mask = self.process(sublayer, x, mask)
			else:
				x, mask = self.process(layer, x, mask)

		return x, mask

	# ----------------------------------------------------------------------------------------------
	def process(self, layer: torch.nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
		if isinstance(layer, torch.nn.Linear):
			x = x.mean(dim=(3, 4)) # NCDHW -> NCD
			x = x.permute(0, 2, 1) # NCD -> NDC
			x = layer(x)
			x = x.permute(0, 2, 1) # NDC -> NCD
		elif isinstance(layer, (Decay3dPartial, PartialConv2d, PartialConv3d, Conv, Convs)):
		# elif isinstance(layer, (Decay3dPartial, PartialConv2d, PartialConv3d, Conv, Convs, UpsampleMask)):
			if mask is not None:
				x, mask = layer(x, mask)
		else:
			x = layer(x)

		return x, mask

	# ----------------------------------------------------------------------------------------------
	def transfer_batch_to_device(self, batch, device, dataloader_idx):
		if isinstance(batch, dict):
			for key in batch:
				if key not in self.cpu_only:
					batch[key] = super().transfer_batch_to_device(batch[key], device, dataloader_idx)
			for key in self.cpu_only:
				if key not in batch:
					warning(f"Model `cpu_only` key '{key}' not found in batch")
			return batch
		else:
			return super().transfer_batch_to_device(batch, device, dataloader_idx)
