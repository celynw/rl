#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack#, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, WarpFrame, ClipRewardEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from rich import print, inspect
from stable_baselines3.common.policies import ActorCriticPolicy as SB3_ACP
from stable_baselines3 import PPO as SB3_PPO

import rl
import rl.models.utils
import rl.models
import rl.utils
from rl.environments.utils import SkipCutscenesPong, VecVideoRecorder, get_base_envs

from enum import Enum, auto

# ==================================================================================================
class EnvType(Enum):
	CARTPOLE = auto()
	MOUNTAINCAR = auto(),
	PONG = auto(),
# ==================================================================================================
class FeatEx(Enum):
	NATURECNN = auto(),
	NATURECNNEVENTS = auto(),
	SNN = auto(),
	EDENN = auto(),

envtype = EnvType.CARTPOLE
featex = FeatEx.EDENN


# ==================================================================================================
def main(args: argparse.Namespace):
	logdir = Path(__file__).parent.resolve()
	config = {
		"env_name": env_name,
		"num_envs": args.n_envs,
		"total_timesteps": int(10e6),
		"seed": 4089164106,
	}

	if not args.nolog:
		run = wandb.init(
			project=args.project,
			name=args.name,
			config=config,
			sync_tensorboard=True, # Auto-upload tensorboard metrics to wandb
			monitor_gym=True, # Auto-upload the videos of agents playing the game
			save_code=True, # Save the code to W&B
			dir=logdir,
		)
		run.log_code(Path(__file__).parent.resolve())
		run.log_code((Path(__file__).parent.parent / "rl").resolve())

	env = make_vec_env(
		env_id=config["env_name"],
		n_envs=config["num_envs"],
		# seed=config["seed"],
		start_index=0,
		monitor_dir=None,
		wrapper_class=wrapper_class,
		env_kwargs=env_kwargs,
		vec_env_cls=None,
		vec_env_kwargs=None,
		monitor_kwargs=None,
	)

	if not args.nolog and not args.novid:
		# FIX: I think I want this to only run on the evaluation
		# With multiple videos, name_prefix needs to match r".+(video\.\d+).+", otherwise they would all conflict under the key "videos" in wandb
		if featex not in [FeatEx.NATURECNN, FeatEx.NATURECNNEVENTS]: # events
			env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 20000 == 0, video_length=video_length, render_events=True, name_prefix="events-video.1")
		if featex is FeatEx.NATURECNNEVENTS: # event images
			env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 20000 == 0, video_length=video_length, render_events=True, sum_events=False, name_prefix="events-video.1")
		env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 20000 == 0, video_length=video_length, name_prefix="rgb-video.2") # RGB

	if envtype is EnvType.PONG:
		env = VecFrameStack(env, n_stack=4)
		# env = VecFrameStack(env, n_stack=4, channels_order="first")
		env = SkipCutscenesPong(env)

	# optimizer_kwargs = dict(
	# )

	# FIX # result_dim = env.state_space.shape[-1] if hasattr(env, "state_space") else env.observation_space.shape[0]
	# result_dim = 4 # CartPole
	features_extractor_kwargs = dict(
		# features_dim=result_dim,
		# features_dim=256,
		features_dim=512,
		#
		# features_dim=256,
		# projection_head=result_dim,
		#
		# # DEBUG
		# features_dim=result_dim,
		# projection_head=result_dim,
	)
	if featex is FeatEx.SNN:
		features_extractor_kwargs["fps"] = get_base_envs(env)[0].fps
		features_extractor_kwargs["tsamples"] = args.tsamples

	# https://github.com/DLR-RM/rl-trained-agents/blob/10a9c31e806820d59b20d8b85ca67090338ea912/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
	# model = SB3_PPO(
	model = rl.models.utils.PPO(
	# model = rl.models.utils.PPO_(
		# policy="CnnPolicy",
		policy=rl.models.utils.ActorCriticPolicy,
		# policy=SB3_ACP,
		# policy_kwargs=dict(features_extractor_class=rl.models.EDeNN, features_extractor_kwargs=features_extractor_kwargs, detach=False),
		# try:
		# - detach=True
		# - optimizer_kwargs=optimizer_kwargs
		policy_kwargs=dict(features_extractor_class=features_extractor_class, net_arch=[], features_extractor_kwargs=features_extractor_kwargs, detach=False),
		env=env,
		batch_size=128, # Probably doesn't do anything, batch size is 1
		clip_range=0.1,
		ent_coef=0.01,
		gae_lambda=0.95,
		gamma=0.99,
		learning_rate=args.lr,
		# learning_rate=lambda f : f * 2.5e-4,
		max_grad_norm=0.5,
		n_epochs=4,
		n_steps=args.n_steps,
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
				save_replay_buffer=True,
				save_vecnormalize=True,
			),
			WandbCallback(
				gradient_save_freq=1000,
				model_save_path=f"models/{run.id}",
			),
		]
	callbacks += [EvalCallback(env, eval_freq=args.n_steps * 10, best_model_save_path=(Path(run.dir) / "checkpoints") / "best" if not args.nolog else None)]
	model.learn(total_timesteps=config["total_timesteps"], callback=callbacks, tb_log_name="tensorboard")
	if not args.nolog:
		model.save(logdir / f"{args.project}_{args.project}.zip")


# ==================================================================================================
class WarpFrame_(WarpFrame):
	"""
	WarpFrame wants HWC instead of CHW.
	Convert to grayscale and warp frames to 84x84 (default)
	as done in the Nature paper and later work.

	:param env: the environment
	:param width:
	:param height:
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
		super().__init__(env, width, height)
		self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.height, self.width), dtype=env.observation_space.dtype)

	# ----------------------------------------------------------------------------------------------
	def observation(self, frame: np.ndarray) -> np.ndarray:
		"""
		returns the current observation from a frame

		:param frame: environment frame
		:return: the observation
		"""
		# frame = np.transpose(frame, (1, 2, 0))
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

		return frame[None, :, :]


# ==================================================================================================
class AtariWrapper(gym.Wrapper):
	"""
	Atari 2600 preprocessings

	Specifically:

	* NoopReset: obtain initial state by taking random number of no-ops on reset.
	* Frame skipping: 4 by default
	* Max-pooling: most recent two observations
	* Termination signal when a life is lost.
	* Resize to a square image: 84x84 by default
	* Grayscale observation
	* Clip reward to {-1, 0, 1}

	:param env: gym environment
	:param noop_max: max number of no-ops
	:param frame_skip: the frequency at which the agent experiences the game.
	:param screen_size: resize Atari frame
	:param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
	:param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(
		self,
		env: gym.Env,
		noop_max: int = 30,
		frame_skip: int = 4,
		screen_size: int = 84,
		terminal_on_life_loss: bool = True,
		clip_reward: bool = True,
	) -> None:
		if noop_max > 0:
			env = NoopResetEnv(env, noop_max=noop_max)
		if frame_skip > 0:
			env = MaxAndSkipEnv(env, skip=frame_skip)
		# if terminal_on_life_loss:
		# 	env = EpisodicLifeEnv(env)
		# if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
		# 	env = FireResetEnv(env)
		env = WarpFrame(env, width=screen_size, height=screen_size)
		# env = WarpFrame(env, width=screen_size, height=screen_size) if featex is FeatEx.NATURECNN else WarpFrame_(env, width=screen_size, height=screen_size)
		# env = WarpFrame_(env, width=screen_size, height=screen_size)
		# env = WarpFrame_(env, width=screen_size, height=screen_size) if featex is FeatEx.NATURECNN else WarpFrame__(env, width=screen_size, height=screen_size)
		if clip_reward:
			env = ClipRewardEnv(env)

		super().__init__(env)


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("project", type=str, help="Name for the wandb project")
	parser.add_argument("name", type=str, help="Name for the wandb run")
	parser.add_argument("--nolog", action="store_true", help="Don't log to wandb")
	parser.add_argument("--novid", action="store_true", help="Don't log videos")
	parser.add_argument("--fps", type=int, default=30)
	parser.add_argument("--tsamples", type=int, default=6)
	parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
	parser.add_argument("-n", "--n_envs", type=int, default=8, help="Number of parallel environments")
	parser.add_argument("--n_steps", type=int, default=128, help="Number of steps before each weights update")
	parser.add_argument("--map_ram", action="store_true", help="Use RAM mappings rather than full RAM state")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	# # DEBUG
	# import random
	# random.seed(0)
	# import torch
	# torch.manual_seed(0)
	# torch.use_deterministic_algorithms(True, warn_only=True)
	# import numpy as np
	# np.random.seed(0)

	args = parse_args()

	if envtype is EnvType.CARTPOLE:
		env_name = "CartPoleRGB-v0" if featex is FeatEx.NATURECNN else "CartPoleEvents-v0"
		output_width = 600
		output_height = 400
	elif envtype is EnvType.MOUNTAINCAR:
		env_name = "MountainCarRGB-v0" if featex is FeatEx.NATURECNN else "MountainCarEvents-v0"
	elif envtype is EnvType.PONG:
		env_name = "PongRGB-v0" if featex is FeatEx.NATURECNN else "PongEvents-v0"

	# wrapper_class = AtariWrapper if envtype is EnvType.PONG and featex is FeatEx.NATURECNN else None
	wrapper_class = AtariWrapper if envtype is EnvType.PONG else None

	env_kwargs = dict(args=args)
	if featex is FeatEx.NATURECNNEVENTS:
		env_kwargs["event_image"] = True

	match featex:
		case FeatEx.EDENN:
			features_extractor_class = rl.models.EDeNN
		case FeatEx.NATURECNN | FeatEx.NATURECNNEVENTS:
			features_extractor_class = rl.models.NatureCNN
		case FeatEx.SNN:
			features_extractor_class = rl.models.SNN

	video_length = 500
	# if envtype is EnvType.CARTPOLE:

	main(args)
