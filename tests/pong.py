#!/usr/bin/env python3
import argparse
from pathlib import Path
# from collections import deque

# import envpool
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage#, VecVideoRecorder
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
	CARTPOLE = auto(),
	PENDULUM = auto(),
	MOUNTAINCAR = auto(),
	PONG = auto(),
	MSPACMAN = auto(),
	MYPACMAN = auto(),
# ==================================================================================================
class FeatEx(Enum):
	NATURECNNRGB = auto(),
	NATURECNNEVENTS = auto(),
	SNN = auto(),
	EDENN = auto(),


# ==================================================================================================
def main(args: argparse.Namespace):
	logdir = Path(__file__).parent.resolve()
	config = {
		"env_name": env_name,
		"num_envs": args.n_envs,
		"total_timesteps": int(10e6),
		# "seed": 4089164106,
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
			resume="allow" if args.resume else "never",
			id=args.load if args.resume else None,
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
	# env = envpool.make(
	# 	task_id=config["env_name"], # v5??
	# 	env_type="gymnasium",
	# 	num_envs=config["num_envs"],
	# 	# seed=config["seed"],
	# 	max_episode_steps=108_000, # Pong (Atari)
	# 	**env_kwargs,
	# )
	# # envs.num_envs = args.num_envs
	# # envs.single_action_space = envs.action_space
	# # envs.single_observation_space = envs.observation_space
	# # envs = RecordEpisodeStatistics(envs)
	# # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
	# env = wrapper_class(env)

	if not args.nolog and not args.novid:
		# FIX: I think I want this to only run on the evaluation
		# With multiple videos, name_prefix needs to match r".+(video\.\d+).+", otherwise they would all conflict under the key "videos" in wandb
		if featex not in [FeatEx.NATURECNNRGB, FeatEx.NATURECNNEVENTS]: # events
			env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 200000 == 0, video_length=video_length, render_events=True, name_prefix="events-video.1")
		if featex is FeatEx.NATURECNNEVENTS: # event images
			env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 200000 == 0, video_length=video_length, render_events=True, sum_events=False, name_prefix="events-video.1")
		env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 200000 == 0, video_length=video_length, name_prefix="rgb-video.2") # RGB
		# TODO add visualisation videos, maybe in separate test script though

	if featex is FeatEx.NATURECNNRGB:
	# if envtype is EnvType.PONG:
		# env = SkipCutscenesPong(env)
		env = VecFrameStack(env, n_stack=4)
		# env = VecFrameStack(env, n_stack=4, channels_order="first")
		# env = VecTransposeImage(env) # Used to stop it complaining that train and eval envs are different (it is auto applied to one of them if not there) - for PongNoFrameSkip-v4?

	# optimizer_kwargs = dict(
	# )

	# FIX # result_dim = env.state_space.shape[-1] if hasattr(env, "state_space") else env.observation_space.shape[0]
	features_extractor_kwargs = dict(
		# features_dim=result_dim,
		# features_dim=256,
		features_dim=result_dim if (args.bs and not args.ph) else 512,
		#
		# features_dim=256,
		# projection_head=result_dim if args.ph else None,
		#
		# # DEBUG
		# features_dim=result_dim,
		# projection_head=result_dim,
	)
	if featex is FeatEx.SNN:
		features_extractor_kwargs["fps"] = get_base_envs(env)[0].fps
		features_extractor_kwargs["tsamples"] = args.tsamples
	elif featex is FeatEx.EDENN:
		features_extractor_kwargs["use_bootstrap"] = args.bs
		features_extractor_kwargs["projection_head"] = result_dim if args.ph else None

	# https://github.com/DLR-RM/rl-trained-agents/blob/10a9c31e806820d59b20d8b85ca67090338ea912/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
	# model = SB3_PPO(
	model = rl.models.utils.PPO(
		# policy="CnnPolicy",
		policy=rl.models.utils.ActorCriticPolicy,
		# policy=SB3_ACP,
		# policy_kwargs=dict(features_extractor_class=rl.models.EDeNN, features_extractor_kwargs=features_extractor_kwargs, detach=False),
		# try:
		# - detach=True
		# - optimizer_kwargs=optimizer_kwargs
		policy_kwargs=dict(features_extractor_class=features_extractor_class, net_arch=[], features_extractor_kwargs=features_extractor_kwargs, detach=args.ph or args.bs),
		# policy_kwargs=dict(features_extractor_class=features_extractor_class, net_arch=[], features_extractor_kwargs=features_extractor_kwargs),
		env=env,
		batch_size=batch_size, # 256, # Probably doesn't do anything, batch size is 1
		clip_range=clip_range, # 0.1
		ent_coef=ent_coef, # 0.01
		gae_lambda=gae_lambda, # 0.95
		gamma=gamma, # 0.99
		# learning_rate=lambda f : f * 2.5e-4 if envtype is EnvType.PONG else args.lr,
		learning_rate=lr,
		max_grad_norm=0.5,
		n_epochs=n_epochs, # 4
		n_steps=n_steps, # args.n_steps,
		vf_coef=0.5,
		tensorboard_log=Path(run.dir) if not args.nolog else None, # Will be appended by `tb_log_name`
		verbose=1,
		device="cpu" if args.cpu else "auto",
	)
	# # print([s.shape for s in list(model.policy.parameters())])
	# for name, param in model.policy.named_parameters():
	# 	# print(f"{name}: {param}")
	# 	if name in [
	# 		"action_net.weight",
	# 		"action_net.bias",
	# 		"value_net.weight",
	# 		"value_net.bias",
	# 	# ] or name.startswith("mlp_extractor.policy_net") or name.startswith("mlp_extractor.value_net"):
	# 	]:
	# 		print(name, param.shape)

	# # LOAD known working model
	# print("BEFORE LOADING: POLICY")
	# for name, param in model.policy.named_parameters():
	# 	print(f"{name}: {param.shape}")
	# print("BEFORE LOADING: FEATURE EXTRACTOR")
	# for name, param in model.policy.features_extractor.named_parameters():
	# 	print(f"{name}: {param.shape}")

	# rollout_buffer = model.rollout_buffer
	# model = rl.models.utils.PPO.load(
	# # model = rl.models.utils.PPO_.load(
	# # model = PPO.load(
	# 	"/home/cw0071/dev/python/rl/tests/wandb/run-20230113_010629-5rlwyuni/files/checkpoints/best/best_model.zip", # My PPO and ACP
	# 	# "/home/cw0071/dev/python/rl/tests/wandb/run-20230113_100256-qurqas1s/files/checkpoints/best/best_model.zip", # Vanilla
	# 	# env=env
	# 	# policy="CnnPolicy",
	# 	policy=rl.models.utils.ActorCriticPolicy,
	# 	# policy=SB3_ACP,
	# 	# policy_kwargs=dict(features_extractor_class=NatureCNN),
	# 	# policy_kwargs=dict(features_extractor_class=NatureCNN, detach=False, net_arch=dict(pi=[64, 64], vf=[64, 64]), features_extractor_kwargs={"features_dim": result_dim}),
	# 	# policy_kwargs=dict(features_extractor_class=NatureCNN, detach=False, features_extractor_kwargs={"features_dim": result_dim}),
	# 	policy_kwargs=dict(features_extractor_class=rl.models.EDeNN, features_extractor_kwargs=features_extractor_kwargs, detach=False),
	# 	# policy_kwargs=dict(features_extractor_class=rl.models.EDeNN, features_extractor_kwargs=features_extractor_kwargs, detach=True),
	# 	# policy_kwargs=dict(detach=False),
	# 	env=env,
	# 	batch_size=256, # probably doesn't do anything, batch size is 1
	# 	clip_range=0.1,
	# 	ent_coef=0.01,
	# 	gae_lambda=0.9,
	# 	gamma=0.99,
	# 	learning_rate=2.5e-4,
	# 	max_grad_norm=0.5,
	# 	n_epochs=4,
	# 	n_steps=args.n_steps,
	# 	vf_coef=0.5,
	# 	tensorboard_log=Path(run.dir) if not args.nolog else None, # Will be appended by `tb_log_name`
	# 	verbose=1,
	# )
	# # model.load_replay_buffer("sac_pendulum_buffer")
	# model.rollout_buffer = rollout_buffer

	# # Freeze RL model, but not the feature extractor
	# for name, param in model.policy.named_parameters():
	# 	# print(f"{name}: {param}")
	# 	if name in [
	# 		"action_net.weight",
	# 		"action_net.bias",
	# 		"value_net.weight",
	# 		"value_net.bias",
	# 	# ] or name.startswith("mlp_extractor.policy_net") or name.startswith("mlp_extractor.value_net"):
	# 	]:
	# 		param.requires_grad = False

	# print(f"Original num. params in optimiser: {len(model.policy.optimizer.param_groups[0]['params'])}")
	# elements = model.policy.optimizer.param_groups[0]
	# del elements["params"]
	# model.policy.optimizer = type(model.policy.optimizer)(params=filter(lambda p: p.requires_grad, model.policy.parameters()), **elements)
	# print(f"New num. params in optimiser: {len(model.policy.optimizer.param_groups[0]['params'])}")

	# print("Requires grad:")
	# for name, param in model.policy.named_parameters():
	# 	if param.requires_grad:
	# 		print(f"{name}: {param.shape}")
	# print("Doesn't require grad:")
	# for name, param in model.policy.named_parameters():
	# 	if not param.requires_grad:
	# 		print(f"{name}: {param.shape}")





	# print("Loading from pretrained feature extractor")
	# import torch
	# checkpoint = torch.load("tools/runs/train_estimator_edenn/version_0/checkpoints/epoch=41-step=52500.ckpt")
	# model.policy.features_extractor.load_state_dict(checkpoint["state_dict"])
	# print("loaded")

	# print("Freezing pretrained feature extractor")
	# for name, param in model.policy.named_parameters():
	# 	if name.startswith("features_extractor"):
	# 		param.requires_grad = False
	# print(f"Original num. params in optimiser: {len(model.policy.optimizer.param_groups[0]['params'])}")
	# elements = model.policy.optimizer.param_groups[0]
	# del elements["params"]
	# model.policy.optimizer = type(model.policy.optimizer)(params=filter(lambda p: p.requires_grad, model.policy.parameters()), **elements)
	# print(f"New num. params in optimiser: {len(model.policy.optimizer.param_groups[0]['params'])}")
	# print("Requires grad:")
	# for name, param in model.policy.named_parameters():
	# 	if param.requires_grad:
	# 		print(f"{name}: {param.shape}")
	# print("Doesn't require grad:")
	# for name, param in model.policy.named_parameters():
	# 	if not param.requires_grad:
	# 		print(f"{name}: {param.shape}")



	if args.load is not None:
		print("Downloading model from wandb")
		# entity, project, run_id, model_artifact_name = wandb.api.get_run_info(f"{args.project}/{args.load}")
		print(wandb.api.default_entity)
		# entity, project, run_id, model_artifact_name = wandb.api.get_run_info(wandb.api.default_entity, args.project, args.load)
		info = wandb.api.get_run_info(wandb.api.default_entity, args.project, args.load)
		print(info)

		args.name += "_cntd" # Note: Need to modify existing wandb.run object

		if info["args"][1] != args.model:
			raise TypeError(f"Tried to run model '{args.model}' but loaded from '{info['args'][1]}'")

		from stable_baselines3.common.save_util import load_from_zip_file
		path = Path("checkpoints") / "best_model.zip"
		wandb.restore(str(path), f"{args.project}/{args.load}")
		model.set_parameters(str(path), device="cpu" if args.cpu else "cuda")

		rollout_buffer = model.rollout_buffer
		model = rl.models.utils.PPO.load(
			str(path),
			policy=rl.models.utils.ActorCriticPolicy,
			policy_kwargs=dict(features_extractor_class=features_extractor_class, net_arch=[], features_extractor_kwargs=features_extractor_kwargs, detach=args.ph or args.bs),
			env=env,
			batch_size=batch_size, # 256, # Probably doesn't do anything, batch size is 1
			clip_range=clip_range, # 0.1
			ent_coef=ent_coef, # 0.01
			gae_lambda=gae_lambda, # 0.95
			gamma=gamma, # 0.99
			learning_rate=lr,
			max_grad_norm=0.5,
			n_epochs=n_epochs, # 4
			n_steps=n_steps, # args.n_steps,
			vf_coef=0.5,
			tensorboard_log=Path(run.dir) if not args.nolog else None, # Will be appended by `tb_log_name`
			verbose=1,
			device="cpu" if args.cpu else "auto",
			print_system_info=True,
		)
		# model.load_replay_buffer("sac_pendulum_buffer")
		model.rollout_buffer = rollout_buffer

		data, _, _ = load_from_zip_file(str(path), device="cpu" if args.cpu else "cuda")

		# Some attrs not being loaded, I'll do it myself
		for name in data:
			try:
				setattr(model, name, data[name])
			except Exception as e:
				# What errors recursive_getattr could throw? KeyError, but possible something else too (e.g. if key is an int?)
				# Catch anything for now
				raise ValueError(f"Key {name} is an invalid object name") from e
			finally:
				print(f"  loaded '{name}' ({data[name]})")
		print("Finished loading!")

		if not args.nolog:
			run.name = args.name


	env.reset() # FIX not being forced as above?

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
		wandb.watch(model.policy.features_extractor, log="all", log_graph=True)
	callbacks += [EvalCallback(env, eval_freq=args.n_steps * 10, best_model_save_path=(Path(run.dir) / "checkpoints") if not args.nolog else None)]
	model.learn(
		total_timesteps=config["total_timesteps"],
		callback=callbacks,
		tb_log_name="tensorboard",
		reset_num_timesteps=False,
	)
	if not args.nolog:
		model.save(logdir / f"{args.project}_{args.name}.zip")


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
		frame = np.transpose(frame, (1, 2, 0))
		# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # For PongNoFrameSkip but not PongRGB?
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

		return frame[None, :, :]


# ==================================================================================================
class WarpFrame__(WarpFrame_):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
		super().__init__(env, width, height)
		self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.height, self.width), dtype=env.observation_space.dtype)

	# ----------------------------------------------------------------------------------------------
	def observation(self, frame: np.ndarray) -> np.ndarray:
		# CDHW
		# print(np.unique(frame[0]), np.unique(frame[1]))
		frame = np.sum(frame, 1) # D
		# print(np.unique(frame[0]), np.unique(frame[1]))
		frame = np.sum(frame, 0) # C
		# print(np.unique(frame))
		frame = ((frame / 6.0) / 2) * 255
		# print(np.unique(frame))
		frame = frame.astype(np.uint8)
		# print(np.unique(frame))
		# print(frame.shape)
		# quit(0)

		return frame[None, :, :]



# ==================================================================================================
class WarpFrameEDeNN(gym.ObservationWrapper):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
		gym.ObservationWrapper.__init__(self, env)
		self.width = width
		self.height = height
		# self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype)
		# self.observation_space = spaces.Box(low=0, high=255, shape=(2, 6, 1, self.height, self.width), dtype=env.observation_space.dtype)
		# print(self.observation_space)

	# # ----------------------------------------------------------------------------------------------
	# def observation(self, frame: np.ndarray) -> np.ndarray:
	# 	"""
	# 	returns the current observation from a frame

	# 	:param frame: environment frame
	# 	:return: the observation
	# 	"""

	# 	# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	# 	# frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
	# 	return frame[:, :, None]

	# # ----------------------------------------------------------------------------------------------
	# def observation(self, frame: np.ndarray) -> np.ndarray:
	# 	# CDHW
	# 	# print(np.unique(frame[0]), np.unique(frame[1]))
	# 	frame = np.sum(frame, 1) # D
	# 	# print(np.unique(frame[0]), np.unique(frame[1]))
	# 	frame = np.sum(frame, 0) # C
	# 	# print(np.unique(frame))
	# 	frame = ((frame / 6.0) / 2) * 255
	# 	# print(np.unique(frame))
	# 	frame = frame.astype(np.uint8)
	# 	# print(np.unique(frame))
	# 	# print(frame.shape)
	# 	# quit(0)

	# 	return frame[None, :, :]


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
		# env = WarpFrame(env, width=screen_size, height=screen_size) if featex is FeatEx.NATURECNNRGB else WarpFrame_(env, width=screen_size, height=screen_size)
		# env = WarpFrame_(env, width=screen_size, height=screen_size)
		# env = WarpFrame_(env, width=screen_size, height=screen_size) if featex is FeatEx.NATURECNNRGB else WarpFrame__(env, width=screen_size, height=screen_size)



		# env = WarpFrame(env, width=screen_size, height=screen_size) if env_name == "PongNoFrameskip-v4" else WarpFrame_(env, width=screen_size, height=screen_size) if env_name == "PongRGB-v0" else env
		env = WarpFrame(env, width=screen_size, height=screen_size) if env_name == "PongNoFrameskip-v4" else env
		# FIX check that env.resize() works in all cases



		if clip_reward:
			env = ClipRewardEnv(env)

		super().__init__(env)


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	global envtype
	global featex
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("project", type=str, help="Name for the wandb project", choices=["CartPole2", "Pendulum", "Pong", "MsPacman", "MyPacman"])
	parser.add_argument("model", type=str, help="Model used. Affects which internal environment is used", choices=["NatureCNN_RGB", "NatureCNN_events", "SNN", "EDeNN"])
	parser.add_argument("-N", "--name", type=str, help="Name for the wandb run. Uses `model` if unspecified", default=None)
	parser.add_argument("--nolog", action="store_true", help="Don't log to wandb")
	parser.add_argument("--novid", action="store_true", help="Don't log videos")
	parser.add_argument("--fps", type=int, default=30)
	parser.add_argument("--tsamples", type=int, default=6)
	parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
	parser.add_argument("-n", "--n_envs", type=int, default=8, help="Number of parallel environments")
	parser.add_argument("--n_steps", type=int, default=128, help="Number of steps before each weights update")
	parser.add_argument("--map_ram", action="store_true", help="Use RAM mappings rather than full RAM state")
	parser.add_argument("--ph", action="store_true", help="Use projection head")
	parser.add_argument("--bs", action="store_true", help="Use bootstrap loss")
	parser.add_argument("--load", type=str, metavar="WANDB_ID", help="Load checkpoint from other run", default=None)
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--pp", action="store_true", help="Use path planner-type env")
	parser.add_argument("--resume", action="store_true", help="Resume a run!")

	args = parser.parse_args()

	if args.project == "CartPole2":
		envtype = EnvType.CARTPOLE
	if args.project == "Pendulum":
		envtype = EnvType.PENDULUM
	elif args.project == "Pong":
		envtype = EnvType.PONG
	elif args.project == "MsPacman":
		envtype = EnvType.MSPACMAN
	elif args.project == "MyPacman":
		envtype = EnvType.MYPACMAN

	if args.model == "NatureCNN_RGB":
		featex = FeatEx.NATURECNNRGB
	elif args.model == "NatureCNN_events":
		featex = FeatEx.NATURECNNEVENTS
	elif args.model == "SNN":
		featex = FeatEx.SNN
	elif args.model == "EDeNN":
		featex = FeatEx.EDENN

	if args.model != "EDeNN":
		args.ph = False
		args.bs = False

	if args.name is None:
		args.name = args.model

	assert not (args.ph and not args.bs)
	if envtype is not EnvType.MYPACMAN:
		assert not args.pp

	return args


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
		env_name = "CartPoleRGB-v0" if featex is FeatEx.NATURECNNRGB else "CartPoleEvents-v0"
		output_width = 600
		output_height = 400

		n_steps = 32
		gae_lambda = 0.8
		gamma = 0.98
		n_epochs = 20
		ent_coef = 0.0
		lr = 0.001
		clip_range = 0.2
		batch_size = 256

		result_dim = 4
	elif envtype is EnvType.PENDULUM:
		env_name = "PendulumRGB-v0" if featex is FeatEx.NATURECNNRGB else "PendulumEvents-v0"
		output_width = 500 # ?
		output_height = 500 # ?

		n_steps = 1024
		gae_lambda = 0.95
		gamma = 0.9
		n_epochs = 10
		ent_coef = 0.0
		lr = 1e-3
		clip_range = 0.2

		batch_size = 256
		result_dim = 2
	elif envtype is EnvType.MOUNTAINCAR:
		env_name = "MountainCarRGB-v0" if featex is FeatEx.NATURECNNRGB else "MountainCarEvents-v0"
	elif envtype in [EnvType.PONG, EnvType.MSPACMAN]:
		# env_name = "PongNoFrameskip-v4" if featex is FeatEx.NATURECNNRGB else "PongEvents-v0"

		if envtype is EnvType.PONG:
			env_name = "PongRGB-v0" if featex is FeatEx.NATURECNNRGB else "PongEvents-v0"
		elif envtype is EnvType.MSPACMAN:
			env_name = "MsPacmanRGB-v0" if featex is FeatEx.NATURECNNRGB else "MsPacmanEvents-v0"

		n_steps = 128
		n_epochs = 4
		ent_coef = 0.01
		lr = 2.5e-4
		clip_range = 0.1
		batch_size = 256

		# ?
		gae_lambda = 0.8
		gamma = 0.98

		result_dim = 6
	elif envtype is EnvType.MYPACMAN:
		if args.pp:
			env_name = "MyPacmanRGBpp-v0" if featex is FeatEx.NATURECNNRGB else "MyPacmanEventspp-v0"
		else:
			env_name = "MyPacmanRGB-v0" if featex is FeatEx.NATURECNNRGB else "MyPacmanEvents-v0"

		n_steps = 128
		n_epochs = 4
		ent_coef = 0.01
		lr = 2.5e-4
		clip_range = 0.1
		batch_size = 256

		# ?
		gae_lambda = 0.8
		gamma = 0.98

		result_dim = 6

	# wrapper_class = AtariWrapper if envtype is EnvType.PONG and featex is FeatEx.NATURECNNRGB else None
	wrapper_class = AtariWrapper if envtype in [EnvType.PONG, EnvType.MSPACMAN] else None

	env_kwargs = dict(args=args)
	# env_kwargs = {}
	if featex is FeatEx.NATURECNNEVENTS:
		env_kwargs["event_image"] = True

	match featex:
		case FeatEx.EDENN:
			features_extractor_class = rl.models.EDeNN
		case FeatEx.NATURECNNRGB | FeatEx.NATURECNNEVENTS:
			features_extractor_class = rl.models.NatureCNN
		case FeatEx.SNN:
			features_extractor_class = rl.models.SNN

	video_length = 500
	# if envtype is EnvType.CARTPOLE:

	main(args)
