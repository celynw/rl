import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from gymnasium import spaces
from ale_py.env.gym import AtariEnv as SB3_AtariEnv
from atariari.benchmark.wrapper import ram2label

from rl.environments.utils import AtariEventEnv

# ==================================================================================================
class MsPacmanEvents(AtariEventEnv):
	# FIX X and Y??
	wanted_states: list[str] = [
		"player_x", "player_y",
		"player_direction",
		"player_score",
		"dots_eaten_count",
		"num_lives",
		"enemy_inky_x", "enemy_inky_y",
		"enemy_pinky_x", "enemy_pinky_y",
		"enemy_blinky_x", "enemy_blinky_y",
		"enemy_sue_x", "enemy_sue_y",
		"ghosts_count",
		"fruit_x", "fruit_y",
	] # TODO choose from these
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		# self.output_height -= (35 + 15) # For `self.resize()`
		# Originally 160x210
		super().__init__(*args, game="ms_pacman", output_width=160, output_height=171, **kwargs)
		# NOTE these don't seem to do anything
		# self.ale.setInt("random_seed", args.seed)
		# self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
		# self.ale.setFloat("repeat_action_probability", 0)
		# self.ale.setInt("frame_skip", 4)
		# self.ale.setBool("color_averaging", True)

		# DEBUG replace ghosts with sprites. But it doesn't work very well
		# self.sprite_blinky = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "sprites" / "blinky.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)
		# self.sprite_pinky = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "sprites" / "pinky.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)
		# self.sprite_inky = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "sprites" / "inky.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)
		# self.sprite_sue = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "sprites" / "sue.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)
		# self.sprite_offset_x = -12
		# self.sprite_offset_y = 1

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEventEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# MsPacmanNoFrameskip-v4
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L1
		# https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4/config.yml
		# and refer to https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4/args.yml
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

		# Crop - 160x171
		frame = frame[2:-37, :, :]

		# FIX WOW! ANY MULTIPLICATION LIKE THIS BREAKS THE EVENT SIMULATION!
		# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # For MsPacmanNoFrameSkip but not MsPacmanRGB?
		# frame = (frame[:, :, 0:1] * 0.299) + (frame[:, :, 1:2] * 0.587) + (frame[:, :, 2:3] * 0.114)
		# frame = frame.astype(np.uint8)
		frame = frame[:, :, :1]

		return frame

	# ----------------------------------------------------------------------------------------------
	# DEBUG replace ghosts with sprites. But it doesn't work very well
	# def render(self):
	# 	bgr = super().render()
	# 	state = ram2label("mspacman", self.ale.getRAM())
	# 	bgr = add_transparent_image(bgr, self.sprite_blinky, state["enemy_blinky_x"] + self.sprite_offset_x, state["enemy_blinky_y"] + self.sprite_offset_y)
	# 	bgr = add_transparent_image(bgr, self.sprite_pinky, state["enemy_pinky_x"] + self.sprite_offset_x, state["enemy_pinky_y"] + self.sprite_offset_y)
	# 	bgr = add_transparent_image(bgr, self.sprite_inky, state["enemy_inky_x"] + self.sprite_offset_x, state["enemy_inky_y"] + self.sprite_offset_y)
	# 	bgr = add_transparent_image(bgr, self.sprite_sue, state["enemy_sue_x"] + self.sprite_offset_x, state["enemy_sue_y"] + self.sprite_offset_y)

	# 	return bgr


# ==================================================================================================
def add_transparent_image(background, foreground, x_offset: int = 0, y_offset: int = 0):
	bg_h, bg_w, bg_channels = background.shape
	fg_h, fg_w, fg_channels = foreground.shape

	assert bg_channels == 3, f"Background image should have exactly 3 channels (RGB). found: {bg_channels}"
	assert fg_channels == 4, f"Foreground image should have exactly 4 channels (RGBA). found: {fg_channels}"

	w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
	h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)
	if w < 1 or h < 1: return

	# Clip foreground and background images to the overlapping regions
	bg_x = max(0, x_offset)
	bg_y = max(0, y_offset)
	fg_x = max(0, x_offset * -1)
	fg_y = max(0, y_offset * -1)
	foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
	background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

	# Separate alpha and color channels from the foreground image
	foreground_colors = foreground[:, :, :3]
	alpha_channel = foreground[:, :, 3] / 255 # 0 - 255 => 0.0 - 1.0

	# Construct an alpha_mask that matches the image shape
	# alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
	alpha_mask = alpha_channel[:, :, np.newaxis]

	# Combine the background with the overlay image weighted by alpha
	composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

	# Overwrite the section of the background image that has been updated
	background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

	return background


# ==================================================================================================
class MsPacmanRGB(SB3_AtariEnv):
	# TODO this has lots of overlap with AtariEventEnv...
	# FIX X and Y??
	wanted_states: list[str] = [
		"player_x", "player_y",
		"player_direction",
		"player_score",
		"dots_eaten_count",
		"num_lives",
		"enemy_inky_x", "enemy_inky_y",
		"enemy_pinky_x", "enemy_pinky_y",
		"enemy_blinky_x", "enemy_blinky_y",
		"enemy_sue_x", "enemy_sue_y",
		"ghosts_count",
		"fruit_x", "fruit_y",
	] # TODO choose from these
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
		output_width: int = 160,
		output_height: int = 171,
	):
		if args.fps is not None:
			self.metadata["render_fps"] = args.fps
		# self.output_height -= (35 + 15) # For `self.resize()`
		self.updatedPolicy = False # Used for logging whenever the policy is updated
		self.game = "ms_pacman"
		self.output_width = output_width
		self.output_height = output_height
		super().__init__(
			game=self.game,
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

		# NOTE these don't seem to do anything
		# self.ale.setInt("random_seed", args.seed)
		# self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
		# self.ale.setFloat("repeat_action_probability", 0)
		# self.ale.setInt("frame_skip", 4)
		# self.ale.setBool("color_averaging", True)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEventEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# MsPacmanNoFrameskip-v4
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L1
		# https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4/config.yml
		# and refer to https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4/args.yml
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
			rgb (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Resized image.
		"""
		frame = frame[2:-37, :, :]
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
		# frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
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
