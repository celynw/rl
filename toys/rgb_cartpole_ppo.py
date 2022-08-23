#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import time
from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
from typing import Union

import torch
import gym
from stable_baselines3.common.monitor import Monitor
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from tqdm import tqdm
from rich import print, inspect

import rl
# from rl.models.utils.visual import CartPoleRGBTemp
from rl.models import PPO_mod, A2C_mod, Estimator, EDeNN
from rl.models.utils import ActorCriticPolicy_mod
from rl.utils import TqdmCallback, PolicyUpdateCallback

use_wandb = False

# ==================================================================================================
def main(args: argparse.Namespace) -> None:
	# env_id = "CartPole-contrast-v1"
	env_id = "CartPole-events-v1"
	# env_id = "CartPole-events-debug"
	# env_id = "CartPole-v1"

	name = f"{int(time.time())}" # Epoch
	if args.name:
		name = f"{name} {args.name}"
	args.log_dir /= name
	print(f"Logging to {args.log_dir}")
	args.log_dir.mkdir(parents=True, exist_ok=True)

	config = {
		"policy_type": ActorCriticPolicy_mod,
		"total_timesteps": max(args.n_steps, args.steps),
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

	all_rewards = []
	env_ = gym.make(env_id)

	for i in tqdm(range(args.times)):
		env = Monitor(env_, str(args.log_dir), allow_early_resets=True, info_keywords=("failReason", "updatedPolicy"))
		policy_kwargs = dict(
			# features_extractor_class=Estimator,
			features_extractor_class=EDeNN,
			optimizer_class=torch.optim.Adam,
		)
		model = PPO_mod(
		# model = A2C_mod(
			ActorCriticPolicy_mod,
			env,
			policy_kwargs=policy_kwargs,
			verbose=1,
			learning_rate=args.lr,
			device="cpu" if args.cpu else "auto",
			n_steps=args.n_steps,
			tensorboard_log=args.log_dir,
			# pl_coef=0.0,
			# ent_coef=0.0,
			# vf_coef=0.0,
			# bs_coef=0.0,
		) # total_timesteps will be at least n_steps (2048)

		if args.load_feat:
			assert model.policy.features_extractor is Estimator, "Only use --load_feat with `Estimator` feature extractor"
			checkpoint = torch.load("/code/train_estimator_RL_estimator/219_1e4b47kx/checkpoints/epoch=59-step=75000.ckpt") # Normal sum(1)
			# checkpoint = torch.load("/code/train_estimator_RL_estimator/221_k1db1ttd/checkpoints/epoch=36-step=46250.ckpt") # Binary->255 image
			# model.policy.features_extractor.load_state_dict(checkpoint["state_dict"], strict=False) # Ignore final layer
			model.policy.features_extractor.load_state_dict(checkpoint["state_dict"])
			# FIX TRY LOADING OPTIMIZER STATE DICT
			# load_state_dict(model.policy.optimizer, checkpoint["optimizer_states"][0])

		if args.load_mlp:
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
		if args.freeze:
			for param in model.policy.features_extractor.parameters():
				param.requires_grad = False
			# model.policy.features_extractor.layer1.0.weight.requires_grad = False
			# model.policy.features_extractor.layer1.0.bias.requires_grad = False
			# FIX can't work out how to do this
			# optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

		env.set_model(model.policy.features_extractor)

		if use_wandb:
			wandbCallback = WandbCallback(
				model_save_path=args.log_dir / f"{args.name}",
				gradient_save_freq=100,
			)
			callbacks = [TqdmCallback(), wandbCallback, PolicyUpdateCallback(env)]
		else:
			callbacks = [TqdmCallback(), PolicyUpdateCallback(env)]
		model.learn(
			total_timesteps=max(model.n_steps, args.steps),
			callback=callbacks,
			eval_freq=args.n_steps,
		)
		all_rewards.append(env.get_episode_rewards())

		if args.save and i == 0:
			model.save(args.log_dir / f"{args.name}")

		if args.render and i == args.times - 1:
			obs = env.reset()
			for i in range(1000):
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)
				env.render()
				if done:
					obs = env.reset()
	env.close()
	# run.finish() # wandb?


# ==================================================================================================
# Copied from torch/optim/optimizer.py version 1.11.0
# Suppressing the ValueError!
def load_state_dict(self, state_dict):
	r"""Loads the optimizer state.

	Args:
		state_dict (dict): optimizer state. Should be an object returned
			from a call to :meth:`state_dict`.
	"""
	# deepcopy, to be consistent with module API
	state_dict = deepcopy(state_dict)
	# Validate the state_dict
	groups = self.param_groups
	saved_groups = state_dict["param_groups"]

	if len(groups) != len(saved_groups):
		raise ValueError("loaded state dict has a different number of "
							"parameter groups")
	param_lens = (len(g["params"]) for g in groups)
	saved_lens = (len(g["params"]) for g in saved_groups)
	if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
		print("### SUPPRESSED ValueError ###")
		print("loaded state dict contains a parameter group "
							"that doesn't match the size of optimizer's group")
		# raise ValueError("loaded state dict contains a parameter group "
		# 					"that doesn't match the size of optimizer's group")

	# Update the state
	id_map = {old_id: p for old_id, p in
				zip(chain.from_iterable((g["params"] for g in saved_groups)),
					chain.from_iterable((g["params"] for g in groups)))}

	def cast(param, value):
		r"""Make a deep copy of value, casting all tensors to device of param."""
		if isinstance(value, torch.Tensor):
			# Floating-point types are a bit special here. They are the only ones
			# that are assumed to always match the type of params.
			if param.is_floating_point():
				value = value.to(param.dtype)
			value = value.to(param.device)
			return value
		elif isinstance(value, dict):
			return {k: cast(param, v) for k, v in value.items()}
		elif isinstance(value, container_abcs.Iterable):
			return type(value)(cast(param, v) for v in value)
		else:
			return value

	# Copy state assigned to params (and cast tensors to appropriate types).
	# State that is not assigned to params is copied as is (needed for
	# backward compatibility).
	state = defaultdict(dict)
	for k, v in state_dict["state"].items():
		if k in id_map:
			param = id_map[k]
			state[param] = cast(param, v)
		else:
			state[k] = v

	# Update parameter groups, setting their 'params' value
	def update_group(group, new_group):
		new_group["params"] = group["params"]
		return new_group
	param_groups = [
		update_group(g, ng) for g, ng in zip(groups, saved_groups)]
	self.__setstate__({"state": state, "param_groups": param_groups})


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("times", type=int, help="Times to run")
	parser.add_argument("-r", "--render", action="store_true", help="Render final trained model output")
	parser.add_argument("-s", "--steps", type=int, default=1000, help="How many steps to train for")
	parser.add_argument("-S", "--save", action="store_true", help="Save first trained model")
	parser.add_argument("-n", "--name", type=str, help="Name of experiment")
	parser.add_argument("-d", "--log_dir", type=Path, default=Path("/tmp/gym/"), help="Location of log directory")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
	parser.add_argument("-f", "--freeze", action="store_true", help="Freeze feature extractor weights")
	parser.add_argument("--load_mlp", action="store_true", help="Load weights for the actor/critic")
	parser.add_argument("--load_feat", action="store_true", help="Load weights for the feature extractor")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps before each weights update")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
