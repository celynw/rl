#!/usr/bin/env python3
import argparse
from typing import Optional

import optuna.trial
import optuna.exceptions
import gym
import torch
from stable_baselines3.common.callbacks import EvalCallback
from rich import print, inspect

import rl.models
import rl.models.utils
import rl.environments
from rl.callbacks import PolicyUpdateCallback, TrialEvalCallback
import rl.utils

# ==================================================================================================
class Objective():
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		"""
		Initialise training object, to possibly be instantiate by `optuna`.

		Args:
			args (argparse.Namespace): Full parsed argument list.
		"""
		self.args = args
		self.log_dir = self.args.log_dir # Preserve value in case we want to repeatedly modify it for optuna

	# ----------------------------------------------------------------------------------------------
	def __call__(self, trial: Optional[optuna.trial.Trial] = None) -> Optional[float]:
		"""
		Run training, either normally or as an `optuna` trial.

		Args:
			trial (Optional[optuna.trial.Trial], optional): Current trial object. Defaults to None.

		Raises:
			optuna.exceptions.TrialPruned: If the trial wasn't doing well, comparitively.

		Returns:
			Optional[float]: Last mean reward of this trial, to be stored and plotted with `optuna`.
		"""
		# Set up logger
		print(f"Logging to {self.log_dir}")
		self.log_dir.mkdir(parents=True, exist_ok=True)

		# Set up environment
		features_extractor_class = getattr(rl.models, self.args.model)
		env = gym.make(
			f"{self.args.environment}-v0",
			event_image=features_extractor_class is rl.models.NatureCNN,
			args=self.args,
		)

		# Set up feature extractor
		features_extractor_kwargs = dict(
			features_dim=env.state_space.shape[-1] if hasattr(env, "state_space") else env.observation_space.shape[0],
		)
		if features_extractor_class is rl.models.EDeNN and self.args.projection_head:
			features_extractor_kwargs = dict(
				features_dim=256,
				projection_head=self.args.projection_head,
				projection_dim=env.state_space.shape[-1] if hasattr(env, "state_space") else env.observation_space.shape[0],
			)
		elif features_extractor_class is rl.models.SNN:
			features_extractor_kwargs.update(dict(
				fps=env.fps,
				tsamples=self.args.tsamples,
			))
		policy_kwargs = dict(
			features_extractor_class=features_extractor_class,
			optimizer_class=torch.optim.Adam,
			features_extractor_kwargs=features_extractor_kwargs,
		)

		# Set up RL model
		model = rl.models.utils.PPO(
			rl.models.utils.ActorCriticPolicy,
			env,
			policy_kwargs=policy_kwargs,
			verbose=1,
			learning_rate=self.args.lr,
			device="cpu" if self.args.cpu else "auto",
			n_steps=self.args.n_steps,
			tensorboard_log=self.log_dir,
			gae_lambda=self.args.gae_lambda,
			gamma=self.args.gamma,
			n_epochs=self.args.n_epochs,
			# pl_coef=0.0,
			# ent_coef=0.0,
			# vf_coef=0.0,
			# bs_coef=0.0,
		) # total_timesteps will be AT LEAST self.args.n_steps
		env.set_model(model.policy.features_extractor)

		# Set up evaluation and callbacks
		tb_log_name = f"{rl.utils.datestr()}_{self.args.name}"
		logger_save_path = self.log_dir / f"{tb_log_name}_1" # Based on stable_baselines3.common.utils.configure_logger
		if trial is not None:
			eval_callback = TrialEvalCallback(env, trial, eval_freq=self.args.n_steps * self.args.eval_every, best_model_save_path=logger_save_path)
		else:
			eval_callback = EvalCallback(env, eval_freq=self.args.n_steps * self.args.eval_every, best_model_save_path=logger_save_path)
		callbacks = [eval_callback, PolicyUpdateCallback(env)]

		# Train
		model.learn(
			total_timesteps=max(model.n_steps, self.args.steps),
			callback=callbacks,
			eval_freq=self.args.n_steps,
			tb_log_name=tb_log_name,
			progress_bar=True,
		)
		if trial is not None and eval_callback.is_pruned:
			raise optuna.exceptions.TrialPruned()

		# Spin down
		env.close()

		# For keeping track with optuna
		if trial is not None:
			return eval_callback.last_mean_reward
