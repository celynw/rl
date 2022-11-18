#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ClipRewardEnv
from stable_baselines3.common.callbacks import EvalCallback
import gym
import numpy as np
import cv2
from gym import spaces
from rich import print, inspect

import rl
import rl.models.utils
import rl.models
import rl.utils
from rl.environments.utils import SkipCutscenesPong

# ==================================================================================================
def main(args: argparse.Namespace):
	logdir = Path(__file__).parent.resolve()
	config = {
		# "env_name": "PongNoFrameskip-v4", # Box(0, 255, (84, 84, 4), uint8)
		"env_name": "CartPoleRGB-v0", # Box(0, 255, (84, 84, 4), uint8)
		"num_envs": 8,
		"total_timesteps": int(10e6),
		"seed": 4089164106,
	}

	if not args.nolog:
		run = wandb.init(
			project=args.project,
			name=args.name,
			config=config,
			sync_tensorboard=True, # Auto-upload tensorboard metrics to wandb
			# monitor_gym=True, # Auto-upload the videos of agents playing the game
			save_code=True, # Save the code to W&B
			dir=logdir,
		)
		run.log_code(Path(__file__).parent.resolve())

	# There already exists an environment generator
	# that will make and wrap atari environments correctly.
	# Here we are also multi-worker training (n_envs=8 => 8 environments)
	# env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"])
	env = make_vec_env(
		env_id=config["env_name"],
		n_envs=config["num_envs"],
		seed=config["seed"],
		start_index=0,
		monitor_dir=None,
		wrapper_class=AtariWrapper,
		env_kwargs=dict(args=args),
		vec_env_cls=None,
		vec_env_kwargs=None,
		monitor_kwargs=None,
	)
	# env = SkipCutscenesPong(env)
	env = VecFrameStack(env, n_stack=4)
	# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)

	# https://github.com/DLR-RM/rl-trained-agents/blob/10a9c31e806820d59b20d8b85ca67090338ea912/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
	n_steps = 128
	model = PPO(
	# model = rl.models.utils.PPO(
		policy="CnnPolicy",
		# policy=rl.models.utils.ActorCriticPolicy,
		# policy_kwargs=dict(features_extractor_class=NatureCNN),
		policy_kwargs=dict(features_extractor_class=rl.models.NatureCNN, net_arch=[]),
		env=env,
		batch_size=256,
		clip_range=0.1,
		ent_coef=0.01,
		gae_lambda=0.9,
		gamma=0.99,
		learning_rate=2.5e-4,
		max_grad_norm=0.5,
		n_epochs=4,
		n_steps=n_steps,
		vf_coef=0.5,
		tensorboard_log=Path(run.dir) if not args.nolog else None, # Will be appended by `tb_log_name`
		verbose=1,
	)

	callbacks = []
	if not args.nolog:
		callbacks += [
			CheckpointCallback(
				save_freq=10000,
				save_path=Path(run.dir) / "checkpoints",
				name_prefix=config["env_name"],
			),
			WandbCallback(
				gradient_save_freq=1000,
				model_save_path=f"models/{run.id}",
			),
		]
	callbacks += [EvalCallback(env, eval_freq=n_steps * 10, best_model_save_path=(Path(run.dir) / "checkpoints") / "best" if not args.nolog else None)]
	model.learn(total_timesteps=config["total_timesteps"], callback=callbacks, tb_log_name="tensorboard")
	if not args.nolog:
		model.save(logdir / f"{args.project}_{args.project}.zip")


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("project", type=str, help="Name for the wandb project")
	parser.add_argument("name", type=str, help="Name for the wandb run")
	parser.add_argument("--nolog", action="store_true", help="Don't log to wandb")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
