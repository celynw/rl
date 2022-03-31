#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
from contextlib import suppress
with suppress(ImportError): import colored_traceback.auto
from typing import Optional

import pytorch_lightning as pl
import optuna
from kellog import info, warning, error, debug
from rich import print, inspect

import rl
from rl import utils
from rl.utils import Step

# ==================================================================================================
class Objective(object):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.logger = utils.setup_logger(self.args)
		info(f"Called command:\n{utils.get_called_command()}")

	# ----------------------------------------------------------------------------------------------
	def __call__(self, trial: Optional[optuna.trial.Trial] = None):
		trainer = pl.Trainer(
			logger=self.logger,
			gpus=0 if self.args.cpu else -1,
			callbacks=utils.setup_callbacks(self.logger, trial),
			limit_train_batches=args.train_lim,
			limit_val_batches=args.val_lim,
			overfit_batches=args.overfit,
			# resume_from_checkpoint=utils.get_checkpoint(self.logger),
			resume_from_checkpoint=utils.get_checkpoint(self.logger, args.checkpoint),
			enable_checkpointing=self.logger is not None,
			max_epochs=1 if args.profile else None,
			profiler="pytorch" if args.profile else None,
			num_sanity_val_steps=0 if args.optuna else 2,
			# For ModelTest, ModelTest2
			gradient_clip_val=0.5,
		)
		Model = getattr(rl.models, self.args.model)
		# Model = rl2.models.DQN
		# Model = ModelTest2

		# import gym
		# from rl.common.vec_env import make_vec_env, SubprocVecEnv
		# env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
		# eval_env = gym.make("CartPole-v1")
		model = Model(self.args, trial=trial)
		trainer.fit(model)
		# model.evaluate(num_eval_episodes=20, render=True)

		# return trainer.callback_metrics[f"loss/{Step.VAL}"].item()
		return trainer.callback_metrics[f"loss/{Step.TRAIN}"].item()


# ==================================================================================================
if __name__ == "__main__":
	args = utils.parse_args()
	if args.seed is not None:
		pl.utilities.seed.seed_everything(seed=args.seed, workers=True)
	if args.optuna:
		try:
			optuna.study.delete_study(study_name="debug", storage="sqlite:///optuna.db")
		except KeyError:
			pass
		study = optuna.create_study(study_name="debug", direction="minimize", storage="sqlite:///optuna.db", load_if_exists=False)
		# study.optimize(main, n_trials=100, n_jobs=1)
		study.optimize(Objective(args), n_trials=100, n_jobs=1)
		study.best_params # E.g. {'x': 2.002108042}
	else:
		Objective(args)()
