"""
Based on AtariEnv from gym=0.25.1
"""
import argparse
from typing import Optional

import numpy as np
import torch
from gym import spaces
from ale_py.env.gym import AtariEnv as SB3_AtariEnv
from atariari.benchmark.wrapper import ram2label

from rl.environments.utils import EventEnv

# ==================================================================================================
class AtariEnv(EventEnv, SB3_AtariEnv):
	state_space: spaces.Space
	wanted_states: list[str] = [] # Placeholder, to be overridden in child classes
	events_width: int = 160 # self.ale.getScreenDims()[1]
	events_height: int = 210 # self.ale.getScreenDims()[0]
	# ----------------------------------------------------------------------------------------------
	def __init__(self, game: str, args: argparse.Namespace, event_image: bool = False,
		frameskip: int | tuple[int, int] = (2, 5), repeat_action_probability: float = 0.0,
		full_action_space: bool = False, max_num_frames_per_episode: int = 108_000,
		return_rgb: bool = False):
		"""
		Event version of Atari environment.

		Args:
			game (str): _description_
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
			event_image (bool, optional): Accuumlates events into an event image. Defaults to False.
			frameskip (int | tuple[int, int], optional): Stochastic frameskip as tuple or fixed. Defaults to (2, 5).
			repeat_action_probability (float, optional): Probability to repeat actions, see Machado et al., 2018. Defaults to 0.0.
			full_action_space (bool, optional): Use full action space? Defaults to False.
			max_num_frames_per_episode (int, optional): Max number of frame per epsiode. Once `max_num_frames_per_episode` is reached the episode is truncated. Defaults to 108_000.
			return_rgb (bool, optional): _description_. Defaults to False.
		"""
		self.game = game
		self.return_rgb = return_rgb
		SB3_AtariEnv.__init__(self, game=game, render_mode="rgb_array",
			frameskip=frameskip,
			repeat_action_probability=repeat_action_probability,
			full_action_space=full_action_space,
			max_num_frames_per_episode=max_num_frames_per_episode,
		)
		self.metadata = { # Compatibility
			# Not sure about this??
			# FIX variable frame_skip will change this too...
			# "render_fps": 30 / float(frameskip)
			"render_fps": 30
		}
		try:
			EventEnv.__init__(self, self.events_width, self.events_height, args, event_image) # type: ignore
		except AttributeError:
			# Cannot override self.observation_space! Grr...
			self._obs_space = spaces.Box(low=0, high=1, shape=self.shape, dtype=np.double)
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
		parser = EventEnv.add_argparse_args(parser)

		group = parser.add_argument_group("Environment")
		group.add_argument("--map_ram", action="store_true", help="Use RAM mappings rather than full RAM state")

		return parser


	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
		"""
		Resets the environment, and also the model (if defined).

		Args:
			seed (int, optional): The seed that is used to initialize the environment's PRNG. Defaults to None.
			options (dict, optional): Additional information to specify how the environment is reset. Defaults to None.

		Returns:
			tuple[np.ndarray, Optional[dict]]: First observation and optionally info about the step.
		"""
		output = super().reset(seed=seed, options=options) # NOTE: Not using the output
		ram = self.ale.getRAM()
		if self.map_ram:
			state = ram2label(self.game, ram)
			state = torch.tensor([state[l] for l in self.wanted_states], dtype=float)
			self.state = state
		else:
			self.state = torch.tensor(ram)
		self.state = (self.state / 128.0) - 1

		return output

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
		events = self.observe(observation)
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

		info = super().get_info()

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return events.numpy(), reward, terminated, truncated, info
