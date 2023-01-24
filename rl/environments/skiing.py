import argparse

import numpy as np

from rl.environments.utils import AtariEventEnv

# ==================================================================================================
class SkiingEvents(AtariEventEnv):
	# FIX object_y=range(87, 94) # object_y_1 is y position of whatever topmost object on the screen is
	wanted_states: list[str] = ["player_x"]
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		super().__init__(*args, game="skiing", **kwargs)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = AtariEventEnv.add_argparse_args(parser)

		# group = [g for g in parser._action_groups if g.title == "Environment"][0]

		# PongNoFrameskip-v4
		# NOTE: They don't have one for skiing
		# NOTE: I'm using a frameskip
		# NOTE: They also use a frame stack of 4
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
	def resize(self, rgb: np.ndarray) -> np.ndarray:
		return rgb
