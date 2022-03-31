import argparse
import os
from typing import Optional
import platform

import torch
import pytorch_lightning as pl
from kellog import info, warning, error, debug
import numpy as np
import optuna

# from rl2.datasets import Base as Dataset
from rl import utils
from rl.utils import Step

# ==================================================================================================
class Base(pl.LightningModule):
	"""
	Base class for ALL MODELS.

	Inherited classes:
	- must implement `step()`, `get_layers()`, `test_epoch_end()`.
	- should reimplement `forward()`.
	- should define `self.datasetType`.
	- can override `self.criterion`.
	- can define `self.metric_XXX` etc..
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None):
		super().__init__()
		self.args = args
		self.trial = trial
		np.set_printoptions(precision=3, linewidth=os.get_terminal_size().columns)
		self.dataModule = utils.DataModule()
		self.layers = self.get_layers() # type:ignore

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		warning("Not overriding base with dataset-specific args")
		# parser = Dataset.add_argparse_args(parser)

		group = parser.add_argument_group("Model")
		# group.add_argument("--ts", type=float, help="Time-discretization in seconds", default=1e-3)
		# group.add_argument("--tsamples", type=int, help="Number of simulation steps", default=100)

		return parser

	# ----------------------------------------------------------------------------------------------
	def on_train_start(self):
		if self.logger:
		 	# For attaching logged loss to hparams
			self.logger.log_hyperparams(self.args, {
				f"loss/{Step.TRAIN}": 0,
				f"loss/{Step.VAL}": 0,
				f"loss/{Step.TEST}": 0,
			})

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
		optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, amsgrad=True)
		lr_scheduler = {
			"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True),
			"monitor": f"loss/{Step.TRAIN}",
		}

		return [optimizer], [lr_scheduler]

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor):
		"""Forward pass."""
		for layer in self.layers:
			x = layer(x)

		return x

	# ----------------------------------------------------------------------------------------------
	def process(self, layer: torch.nn.Module, x: torch.Tensor):
		if isinstance(layer, torch.nn.Linear):
			x = x.mean(dim=(3, 4)) # NCDHW -> NCD
			x = x.permute(0, 2, 1) # NCD -> NDC
			x = layer(x)
			x = x.permute(0, 2, 1) # NDC -> NCD
		else:
			x = layer(x)

		return x
