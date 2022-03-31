#!/usr/bin/env python3
import argparse

import torch
from torch.utils.data import DataLoader

from kellog import info, warning, error, debug

from rl.models import Base
from rl.utils import Step
from rl.datasets import GymEnv

# ==================================================================================================
class RL(Base):
	"""
	Base class for reinforcement learning.

	Inherited classes:
	- must implement `step()`, `get_layers()`.
	- must set `env_type: str`, `sb3_model`.
	- should reimplement `forward()`.
	- should extend `test_step()`.
	- should define `self.env_type`.
	- can override `self.criterion`.
	- can define `self.metric_XXX` etc..
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial):
		super().__init__(args, trial)
		self.layers = self.get_layers() # type:ignore
		self.criterion = torch.nn.MSELoss(reduction="none")
		# self.example_input_array = torch.zeros((1, 2, 100, self.datasetType.height, self.datasetType.width)).float()
		# if self.datasetType in [EVReflex, MVSEC]:
		# 	self.cpu_only = [
		# 		"imgs",
		# 		"stamps",
		# 	]
		# self.dataModule = Env(self.env_type, self.args, step)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = GymEnv.add_argparse_args(parser)

		return parser


	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch, batch_idx):
		return self.step(Step.TRAIN, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def validation_step(self, batch, batch_idx):
		return self.step(Step.VAL, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def test_step(self, batch, batch_idx):
		return self.step(Step.TEST, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def dataloader(self, step: Step) -> DataLoader:
		print(self.args)
		return DataLoader(
			# self.datasetType(self.args, step),
			# GymEnv(self.args, step)
			GymEnv(self.args, self.env_type, self.sb3_model(self.env_type)),
			# batch_size=(1 if step == Step.TEST else self.args.batch_size),
			batch_size=1, # TODO
			# shuffle=(step == Step.TRAIN and self.args.overfit == 0),
			shuffle=False, # TODO
			# num_workers=self.args.workers,
			num_workers=0, # TODO
			# pin_memory=self.device != torch.device("cpu"),
			pin_memory=False, # TODO
			# persistent_workers=self.args.workers > 0,
			persistent_workers=False, # TODO
			# drop_last=(step != Step.TEST),
			drop_last=False,
		)

	# # ==================================================================================================
	# class OnPolicyDataloader:
	# 	# ----------------------------------------------------------------------------------------------
	# 	def __init__(self, model: OnPolicyModel):
	# 		self.model = model

	# 	# ----------------------------------------------------------------------------------------------
	# 	def __iter__(self):
	# 		for i in range(self.model.num_rollouts):
	# 			experiences = self.model.collect_rollouts()
	# 			observations, actions, old_values, old_log_probs, advantages, returns = experiences
	# 			for j in range(self.model.epochs_per_rollout):
	# 				k = 0
	# 				perm = torch.randperm(observations.shape[0], device=observations.device)
	# 				while k < observations.shape[0]:
	# 					batch_size = min(observations.shape[0] - k, self.model.batch_size)
	# 					yield RolloutBufferSamples(
	# 						observations[perm[k:k+batch_size]],
	# 						actions[perm[k:k+batch_size]],
	# 						old_values[perm[k:k+batch_size]],
	# 						old_log_probs[perm[k:k+batch_size]],
	# 						advantages[perm[k:k+batch_size]],
	# 						returns[perm[k:k+batch_size]])
	# 					k += batch_size

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self) -> DataLoader:
		return self.dataloader(Step.TRAIN)

	# ----------------------------------------------------------------------------------------------
	def val_dataloader(self) -> DataLoader:
		return self.dataloader(Step.VAL)

	# ----------------------------------------------------------------------------------------------
	def test_dataloader(self) -> DataLoader:
		return self.dataloader(Step.TEST)
