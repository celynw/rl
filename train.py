#!/usr/bin/env python3
import argparse
from contextlib import suppress
with suppress(ImportError): import colored_traceback.auto
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
import optuna
from kellog import info, warning, error, debug
from rich import print, inspect

import rl
from rl import utils

# ==================================================================================================
class Objective():
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.logger = utils.setup_logger(self.args)
		info(f"Called command:\n{utils.get_called_command()}")

	# ----------------------------------------------------------------------------------------------
	def __call__(self, trial: Optional[optuna.trial.Trial] = None):
		if trial is not None:
			raise NotImplementedError("TODO Suggest hparams in arg parsing")
			import socket
			trial.set_user_attr("host", socket.gethostname())
			trial.set_user_attr("git", utils.get_git_rev())
			trial.set_user_attr("cmd", utils.get_called_command())

		Model = getattr(rl.models, self.args.model)
		checkpointPath = utils.get_checkpoint(self.logger, self.args.checkpoint)
		if checkpointPath is not None and checkpointPath.exists():
			# Neded for loading the original args/hparams as opposed to default argparse args again
			model = Model.load_from_checkpoint(checkpointPath)
			model.hparams.max_epochs = self.args.max_epochs
		else:
			model = Model(self.args, trial)
		trainer = pl.Trainer(
			logger=self.logger,
			gpus=0 if self.args.cpu else -1,
			# resume_from_checkpoint=Path(),
			callbacks=utils.setup_callbacks(self.logger, Model.monitor, Model.monitor_dir, trial),
			accumulate_grad_batches=self.args.batch_accumulation,
			overfit_batches=self.args.overfit,
			enable_checkpointing=self.logger is not None,
			max_epochs=1 if self.args.profile else self.args.max_epochs,
			profiler="pytorch" if self.args.profile else None,
			num_sanity_val_steps=0 if self.args.optuna is not None else 2,
			enable_model_summary=False, # This is done manually in the callbacks
			auto_lr_find=self.args.autoLR,
			# progress_bar_refresh_rate=0 if self.args.optuna is not None else None,
			# val_check_interval=20, # WARNING: increments only within an epoch and resets for new ones! Use ModelCheckpoint's every_n_train_steps instead
			# val_check_interval=self.args.episode_length,
		)
		if self.args.autoLR:
			# trainer.tune(model)
			lr_finder = trainer.tuner.lr_find(model, num_training=20)
			lr_finder.plot(suggest=True).savefig(Path(self.logger.log_dir) / "autoLR.png")
			model.hparams.lr = lr_finder.suggestion()
			model.args.lr = model.hparams.lr
		trainer.fit(model, ckpt_path=checkpointPath) # Needed for continuing from the right epoch
		# trainer.test(ckpt_path="best")

		return trainer.callback_metrics[Model.monitor].item()


# ==================================================================================================
if __name__ == "__main__":
	torch.set_printoptions(precision=16, sci_mode=False)
	args = utils.parse_args()
	if args.seed is not None:
		pl.utilities.seed.seed_everything(seed=args.seed, workers=True)
	if args.optuna is not None:
		study = optuna.create_study(
			study_name=f"{args.model}_{args.dataset}",
			direction=optuna.study.StudyDirection.MINIMIZE,
			storage=args.optuna,
			load_if_exists=True
		)
		optuna.create_study()
		study.optimize(Objective(args), n_trials=100, n_jobs=1, gc_after_trial=False)
		info(f"Best params so far: {study.best_params}")
	else:
		Objective(args)()
