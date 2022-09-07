#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import time
from typing import Union, Optional

import torch
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from rich import print, inspect
import optuna

import rl
# from rl.models.utils.visual import CartPoleRGBTemp
from rl.models import PPO_mod, A2C_mod, Estimator, EstimatorPH, EDeNN, EDeNNPH
from rl.models.utils import ActorCriticPolicy_mod
from rl.utils import TqdmCallback, PolicyUpdateCallback, TrialEvalCallback
# from rl.utils import load_optimizer_state_dict
from rl.utils import MultiplePruners, DelayedThresholdPruner

use_wandb = False

# ==================================================================================================
class Objective():
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.log_dir = self.args.log_dir # Preserve value in case we want to repeatedly modify it for optuna

	# ----------------------------------------------------------------------------------------------
	def __call__(self, trial: Optional[optuna.trial.Trial] = None):
		if trial is not None:
			# raise NotImplementedError("TODO Suggest hparams in arg parsing")
			# self.args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True) # Default was 3e-4
			self.args.tsamples = trial.suggest_int("tsamples", 1, 40) # EDeNN specific TODO what should max be?

		# env_id = "CartPole-contrast-v1"
		env_id = "CartPole-events-v1"
		# env_id = "CartPole-events-debug"
		# env_id = "CartPole-v1"

		if trial is not None:
			# Insert trial number at start of child directory name
			self.log_dir = self.args.log_dir.parent / f"{trial.number} {self.args.log_dir.name}"

		print(f"Logging to {self.args.log_dir}")
		self.log_dir.mkdir(parents=True, exist_ok=True)

		config = {
			"policy_type": ActorCriticPolicy_mod,
			"total_timesteps": max(self.args.n_steps, self.args.steps),
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

		env_ = gym.make(env_id, tsamples=self.args.tsamples)
		env = Monitor(env_, str(self.log_dir), allow_early_resets=True, info_keywords=("failReason", "updatedPolicy"))

		# extractor = EstimatorPH if self.args.projection_head else Estimator
		extractor = EDeNNPH if self.args.projection_head else EDeNN
		if self.args.projection_head:
			features_extractor_kwargs=dict(features_dim=256)
		else:
			features_extractor_kwargs = None
		policy_kwargs = dict(features_extractor_class=extractor, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs)

		model = PPO_mod(
		# model = A2C_mod(
			ActorCriticPolicy_mod,
			env,
			policy_kwargs=policy_kwargs,
			verbose=1,
			learning_rate=self.args.lr,
			device="cpu" if self.args.cpu else "auto",
			n_steps=self.args.n_steps,
			tensorboard_log=self.log_dir,
			# pl_coef=0.0,
			# ent_coef=0.0,
			# vf_coef=0.0,
			# bs_coef=0.0,
		) # total_timesteps will be at least n_steps (2048)

		if self.args.load_feat:
			assert model.policy.features_extractor is Estimator, "Only use --load_feat with `Estimator` feature extractor"
			checkpoint = torch.load("/code/train_estimator_RL_estimator/219_1e4b47kx/checkpoints/epoch=59-step=75000.ckpt") # Normal sum(1)
			# checkpoint = torch.load("/code/train_estimator_RL_estimator/221_k1db1ttd/checkpoints/epoch=36-step=46250.ckpt") # Binary->255 image
			# model.policy.features_extractor.load_state_dict(checkpoint["state_dict"], strict=False) # Ignore final layer
			model.policy.features_extractor.load_state_dict(checkpoint["state_dict"])
			# FIX TRY LOADING OPTIMIZER STATE DICT
			# load_optimizer_state_dict(model.policy.optimizer, checkpoint["optimizer_states"][0])
		if self.args.load_mlp:
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
		if self.args.freeze:
			for param in model.policy.features_extractor.parameters():
				param.requires_grad = False
			# model.policy.features_extractor.layer1.0.weight.requires_grad = False
			# model.policy.features_extractor.layer1.0.bias.requires_grad = False
			# FIX can't work out how to do this
			# optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

		env.set_model(model.policy.features_extractor)

		callbacks = [TqdmCallback(), PolicyUpdateCallback(env)]
		if use_wandb:
			wandbCallback = WandbCallback(
				model_save_path=self.log_dir / f"{self.args.name}",
				gradient_save_freq=100,
			)
			callbacks.append(wandbCallback)
		if trial is not None:
			eval_callback = TrialEvalCallback(env, trial, eval_freq=self.args.n_steps)
			callbacks.append(eval_callback)
		else:
			eval_callback = EvalCallback(env, eval_freq=self.args.n_steps)
			callbacks.append(eval_callback)

		model.learn(
			total_timesteps=max(model.n_steps, self.args.steps),
			callback=callbacks,
			eval_freq=self.args.n_steps,
		)
		if trial is not None and eval_callback.is_pruned:
			raise optuna.exceptions.TrialPruned()
		elif self.args.save:
			model.save(self.log_dir / f"{self.args.name}")

		# if self.args.render and i == self.args.times - 1:
		# 	obs = env.reset()
		# 	for i in range(1000):
		# 		action, _states = model.predict(obs, deterministic=True)
		# 		obs, reward, done, info = env.step(action)
		# 		env.render()
		# 		if done:
		# 			obs = env.reset()

		env.close()
		if use_wandb:
			run.finish()

		if trial is not None:
			return eval_callback.last_mean_reward


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("name", type=str, help="Name of experiment")
	parser.add_argument("-r", "--render", action="store_true", help="Render final trained model output")
	parser.add_argument("-s", "--steps", type=int, default=1000, help="How many steps to train for")
	parser.add_argument("-S", "--save", action="store_true", help="Save first trained model")
	parser.add_argument("-d", "--log_dir", type=Path, default=Path("/tmp/gym/"), help="Location of log directory")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
	parser.add_argument("--tsamples", type=int, default=10, help="Time samples for env, propagates through EDeNN")
	parser.add_argument("-f", "--freeze", action="store_true", help="Freeze feature extractor weights")
	parser.add_argument("--load_mlp", action="store_true", help="Load weights for the actor/critic")
	parser.add_argument("--load_feat", action="store_true", help="Load weights for the feature extractor")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps before each weights update")
	parser.add_argument("--projection_head", action="store_true", help="Use proejction head")
	parser.add_argument("--optuna", type=str, help="Optimise with optuna using this storage URL. Examples: 'sqlite:///optuna.db' or 'postgresql://postgres:password@host:5432/postgres'")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	args = parse_args()
	epoch = f"{int(time.time())}" # Epoch
	if args.optuna is not None:
		args.log_dir = args.log_dir / args.name / epoch
		torch.set_num_threads(1)
		# For RandomSampler, MedianPruner is the best
		# For TPESampler (default), Hyperband is the best
		pruner = MultiplePruners((optuna.pruners.HyperbandPruner(), DelayedThresholdPruner(lower=10, max_prunes=3)))
		study = optuna.create_study(
			study_name=f"{args.name}",
			direction=optuna.study.StudyDirection.MAXIMIZE,
			storage=args.optuna,
			load_if_exists=True,
			pruner=pruner,
		)
		study.optimize(Objective(args), n_trials=100, n_jobs=1, gc_after_trial=False)
		# print(f"Best params so far: {study.best_params}")
		print(f"Number of finished trials: {len(study.trials)}")
		print("Best trial:")
		trial = study.best_trial
		print(f"  Value: {trial.value}")
		print("  Params: ")
		for key, value in trial.params.items():
			print(f"    {key}: {value}")
		print("  User attrs:")
		for key, value in trial.user_attrs.items():
			print(f"    {key}: {value}")
	else:
		args.log_dir /= f"{epoch} {args.name}"
		Objective(args)()
