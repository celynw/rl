import argparse

import numpy as np

from rl.environments.utils import AtariEnv

# ==================================================================================================
class PongEvents(AtariEnv):
	# FIX X and Y??
	wanted_states: list[str] = ["player_y", "player_x", "enemy_y", "enemy_x", "ball_x", "ball_y"]
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		self.output_height -= (35 + 15) # For `self.resize()`
		super().__init__(*args, game="pong", **kwargs)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# PongNoFrameskip-v4
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L1
		# https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
		# and refer to https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/args.yml
		# # parser.set_defaults(steps=10000000)
		parser.set_defaults(n_steps=128 * 8) # n_envs = 8, rollout buffer size is n_steps * n_envs
		parser.set_defaults(n_epochs=4)
		parser.set_defaults(ent_coef=0.01)
		parser.set_defaults(lr=2.5e-4)
		parser.set_defaults(clip_range=0.1)
		parser.set_defaults(batch_size=256)

		return parser

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb: np.ndarray) -> np.ndarray:
		"""
		Crop the top and bottom, then reduce the resolution.

		Args:
			rgb (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Resized image.
		"""
		# SIMON - Crop
		rgb = rgb[35:-15, :, :]
		# SIMON - Naively convert to greyscale (shit way to do it but don't want extra imports)
		rgb[:, :, 0] = rgb[:, :, 0] / 3 + rgb[:, :, 1] / 3 + rgb[:, :, 2] / 3
		# SIMON - Make it 3 channel again
		rgb[:, :, 1] = rgb[:, :, 0]
		rgb[:, :, 2] = rgb[:, :, 0]
		# # SIMON - rescale contrast
		# rgb = rgb - np.min(rgb)
		# rgb = np.clip(rgb * (255 / np.max(rgb)), 0, 255).astype("uint8")
		# SIMON - rescale contrast
		rgb = rgb - 57
		rgb = np.clip(rgb * (255 / 89), 0, 255).astype("uint8")

		return rgb
