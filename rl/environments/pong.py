import argparse
from typing import Optional

import numpy as np
import torch
import cv2
from gymnasium import spaces
from ale_py.env.gym import AtariEnv as SB3_AtariEnv
from atariari.benchmark.wrapper import ram2label

from rl.environments.utils import AtariEventEnv

# ==================================================================================================
class PongEvents(AtariEventEnv):
	# FIX X and Y??
	wanted_states: list[str] = ["player_y", "player_x", "enemy_y", "enemy_x", "ball_x", "ball_y"]
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		# self.output_height -= (35 + 15) # For `self.resize()`
		super().__init__(*args, game="pong", output_width=84, output_height=84, **kwargs)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEventEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# PongNoFrameskip-v4
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L1
		# https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
		# and refer to https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/args.yml
		# # parser.set_defaults(steps=10000000)
		parser.set_defaults(n_envs=8)
		parser.set_defaults(n_steps=128)
		parser.set_defaults(n_epochs=4)
		parser.set_defaults(ent_coef=0.01)
		parser.set_defaults(lr=2.5e-4)
		parser.set_defaults(clip_range=0.1)
		parser.set_defaults(batch_size=256)

		return parser

	# ----------------------------------------------------------------------------------------------
	def resize(self, frame: np.ndarray) -> np.ndarray:
		"""
		Crop the top and bottom, then reduce the resolution.

		Args:
			frame (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Resized image.
		"""
		# frame = super().resize(frame)

		# SIMON - Crop
		# frame = frame[35:-15, :, :]
		# SIMON - Naively convert to greyscale (shit way to do it but don't want extra imports)
		frame[:, :, 0] = frame[:, :, 0] / 3 + frame[:, :, 1] / 3 + frame[:, :, 2] / 3
		# SIMON - Make it 3 channel again
		frame[:, :, 1] = frame[:, :, 0]
		frame[:, :, 2] = frame[:, :, 0]
		# # SIMON - rescale contrast
		# frame = frame - np.min(frame)
		# frame = np.clip(frame * (255 / np.max(frame)), 0, 255).astype("uint8")
		# SIMON - rescale contrast
		frame = frame - 57
		frame = np.clip(frame * (255 / 89), 0, 255).astype("uint8")

		# frame = np.transpose(frame, (1, 2, 0))
		# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # For PongNoFrameSkip but not PongRGB?
		frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)

		return frame



# ==================================================================================================
class PongRGB(SB3_AtariEnv):
	# TODO this has lots of overlap with AtariEventEnv...
	# FIX X and Y??
	wanted_states: list[str] = ["player_y", "player_x", "enemy_y", "enemy_x", "ball_x", "ball_y"]
	# ----------------------------------------------------------------------------------------------
	def __init__(
		self,
		args: argparse.Namespace,
		frameskip: tuple[int, int] | int = 1, # XXXXXXNoFrameSkip-v4
		repeat_action_probability: float = 0.0, # XXXXXX-v4
		full_action_space: bool = False, # XXXXXX-v4
		max_num_frames_per_episode: Optional[int] = 108_000, # XXXXXX-v4
		# output_width: int = 160, # self.ale.getScreenDims()[1]
		# output_height: int = 210, # self.ale.getScreenDims()[0]
		output_width: int = 84,
		output_height: int = 84,
	):
		if args.fps is not None:
			self.metadata["render_fps"] = args.fps
		# self.output_height -= (35 + 15) # For `self.resize()`
		self.updatedPolicy = False # Used for logging whenever the policy is updated
		self.game = "pong"
		self.output_width = output_width
		self.output_height = output_height
		super().__init__(
			game="pong",
			obs_type="rgb", # try "grayscale"?,
			# obs_type="grayscale", # try "grayscale"?,
			frameskip=frameskip,
			repeat_action_probability=repeat_action_probability,
			full_action_space=full_action_space, # XXXXXX-v4
			max_num_frames_per_episode=max_num_frames_per_episode, # XXXXXX-v4
			render_mode="rgb_array",
		)
		image_shape = (output_height, output_width)
		# if self._obs_type == "rgb":
		# 	image_shape += (3,)
		# self._obs_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=image_shape)
		self._obs_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=[1, output_height, output_width])
					# self.shape = [2, height, width]

		self.render_mode = self._render_mode # Stupid compatibility

		self.map_ram = args.map_ram

		if self.map_ram:
			self.state_space = spaces.Space([1, len(self.wanted_states)])
		else:
			self.state_space = spaces.Space([1, 128])

		ram = self.ale.getRAM()
		if self.map_ram:
			state = ram2label(self.game, ram)
			state = torch.tensor([state[l] for l in self.wanted_states], dtype=float)
			self.state = state
		else:
			self.state = torch.tensor(ram)
		self.state = (self.state / 128.0) - 1

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEventEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# PongNoFrameskip-v4
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L1
		# https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
		# and refer to https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/args.yml
		# # parser.set_defaults(steps=10000000)
		parser.set_defaults(n_steps=128) # rollout buffer full size is n_steps * n_envs
		parser.set_defaults(n_epochs=4)
		parser.set_defaults(ent_coef=0.01)
		parser.set_defaults(lr=2.5e-4)
		parser.set_defaults(clip_range=0.1)
		parser.set_defaults(batch_size=256)

		return parser

	# ----------------------------------------------------------------------------------------------
	def resize(self, frame: np.ndarray) -> np.ndarray:
		"""
		Crop the top and bottom, then reduce the resolution.

		Args:
			rgb (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Resized image.
		"""
		# SIMON - Crop
		# rgb = rgb[35:-15, :, :]
		# # SIMON - Naively convert to greyscale (shit way to do it but don't want extra imports)
		# rgb[:, :, 0] = rgb[:, :, 0] / 3 + rgb[:, :, 1] / 3 + rgb[:, :, 2] / 3
		# # SIMON - Make it 3 channel again
		# rgb[:, :, 1] = rgb[:, :, 0]
		# rgb[:, :, 2] = rgb[:, :, 0]
		# # # SIMON - rescale contrast
		# # rgb = rgb - np.min(rgb)
		# # rgb = np.clip(rgb * (255 / np.max(rgb)), 0, 255).astype("uint8")
		# # SIMON - rescale contrast
		# rgb = rgb - 57
		# rgb = np.clip(rgb * (255 / 89), 0, 255).astype("uint8")

		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
		frame = frame[..., None] # Put channel dimension back

		return frame

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict:
		"""
		Return a created dictionary for the step info.

		Returns:
			dict: Key-value pairs for the step info.
		"""
		return {
			"state": self.state, # Used later for bootstrap loss
			"updatedPolicy": int(self.updatedPolicy),
		}

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
		"""
		Resets the environment.

		Args:
			seed (int, optional): The seed that is used to initialize the environment's PRNG. Defaults to None.
			options (dict, optional): Additional information to specify how the environment is reset. Defaults to None.

		Returns:
			tuple[np.ndarray, Optional[dict]]: First observation and optionally info about the step.
		"""
		output, info = super().reset(seed=seed, options=options) # NOTE: Not using the output
		output = self.resize(output)
		output = np.transpose(output, (2, 0, 1)) # HWC -> CHW
		ram = self.ale.getRAM()
		if self.map_ram:
			state = ram2label(self.game, ram)
			state = torch.tensor([state[l] for l in self.wanted_states], dtype=float)
			self.state = state
		else:
			self.state = torch.tensor(ram)
		self.state = (self.state / 128.0) - 1

		return output, info

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
		"""
		Perform a single environment step.

		Args:
			action (int): Which action to perform this step.

		Returns:
			tuple[np.ndarray, float, bool, bool, dict]: Step returns.
		"""
		observation, reward, terminated, truncated, _ = super().step(action) # type: ignore
		observation = self.resize(observation)
		observation = np.transpose(observation, (2, 0, 1)) # HWC -> CHW
		ram = self.ale.getRAM()
		# self.ale.getScreenRGB()
		# self.ale.getScreenGrayscale()
		if self.map_ram:
			state = ram2label(self.game, ram)
			state = torch.tensor([state[l] for l in self.wanted_states], dtype=float)
			self.state = state
		else:
			self.state = torch.tensor(ram)
		self.state = (self.state / 128.0) - 1

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return observation, reward, terminated, truncated, self.get_info()
