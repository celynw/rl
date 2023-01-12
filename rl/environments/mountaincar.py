import argparse
import math
from typing import Optional

from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium import spaces
import numpy as np
import pygame, pygame.gfxdraw
import gymnasium as gym
from rich import print, inspect

from rl.environments.utils import EventEnv
from rl.models import EDeNN, SNN

# ==================================================================================================
class MountainCarEvents(EventEnv, MountainCarEnv):
	state_space: spaces.Space
	model: Optional[EDeNN | SNN] = None
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, event_image: bool = False, return_rgb: bool = False):
		"""
		Event version of MountainCar environment.

		Args:
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
			event_image (bool, optional): Accuumlates events into an event image. Defaults to False.
			return_rgb (bool, optional): _description_. Defaults to False.
		"""
		self.return_rgb = return_rgb
		self.output_width = 150
		self.output_height = 100
		MountainCarEnv.__init__(self, render_mode="rgb_array")
		EventEnv.__init__(self, self.output_width, self.output_height, args, event_image) # type: ignore

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = EventEnv.add_argparse_args(parser)
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L45
		# parser.set_defaults(steps=1e6)
		parser.set_defaults(n_steps=16 * 16) # n_envs = 16, rollout buffer size is n_steps * n_envs
		parser.set_defaults(gae_lambda=0.95)
		parser.set_defaults(gamma=0.99)
		parser.set_defaults(n_epochs=4)
		parser.set_defaults(ent_coef=0.0)

		return parser

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
		"""
		Perform a single environment step.

		Args:
			action (int): Which action to perform this step.

		Returns:
			tuple[np.ndarray, float, bool, bool, dict]: Step returns.
		"""
		_, reward, terminated, truncated, _ = super().step(action) # type: ignore
		events = self.observe()

		info = super().get_info()

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return events.numpy(), reward, terminated, truncated, info

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb: np.ndarray) -> np.ndarray:
		"""
		Don't do anything.

		Args:
			rgb (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Original image.
		"""
		return rgb

	# ----------------------------------------------------------------------------------------------
	def render(self) -> np.ndarray:
		"""
		Copied and adapted from MountainCarEnv.render() [gym==0.26.2].

		Args:
			mode (str, optional): Render mode. Defaults to "rgb_array".

		Returns:
			np.ndarray: Rendered image.
		"""
		if self.render_mode is None:
			gym.logger.warn(
				"You are calling render method without specifying any render mode. "
				"You can specify the render_mode at initialization, "
				f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
			)
			return

		if self.screen is None:
			pygame.init()
			if self.render_mode == "human":
				pygame.display.init()
				self.screen = pygame.display.set_mode(
					(self.screen_width, self.screen_height)
				)
			else: # mode in "rgb_array"
				self.screen = pygame.Surface((self.screen_width, self.screen_height))
		if self.clock is None:
			self.clock = pygame.time.Clock()

		world_width = self.max_position - self.min_position
		scale = self.screen_width / world_width
		carwidth = 20 # NOTE: Changing car dimensions
		carheight = 10 # NOTE: Changing car dimensions

		self.surf = pygame.Surface((self.screen_width, self.screen_height))
		self.surf.fill((255, 255, 255))

		pos = self.state[0]

		xs = np.linspace(self.min_position, self.max_position, 100)
		ys = self._height(xs)
		xys = list(zip((xs - self.min_position) * scale, ys * scale))

		pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

		clearance = 10

		l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
		coords = []
		for c in [(l, b), (l, t), (r, t), (r, b)]:
			c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
			coords.append(
				(
					c[0] + (pos - self.min_position) * scale,
					c[1] + clearance + self._height(pos) * scale,
				)
			)

		pygame.gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
		pygame.gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

		for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
			c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
			wheel = (
				int(c[0] + (pos - self.min_position) * scale),
				int(c[1] + clearance + self._height(pos) * scale),
			)

			# # NOTE: Not drawing the wheels
			pygame.gfxdraw.aacircle(
				self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
			)
			pygame.gfxdraw.filled_circle(
				self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
			)

		flagx = int((self.goal_position - self.min_position) * scale)
		flagy1 = int(self._height(self.goal_position) * scale)
		flagy2 = flagy1 + 50
		pygame.gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

		pygame.gfxdraw.aapolygon(
			self.surf,
			[(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
			(204, 204, 0),
		)
		pygame.gfxdraw.filled_polygon(
			self.surf,
			[(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
			(204, 204, 0),
		)

		self.surf = pygame.transform.flip(self.surf, False, True)
		self.screen.blit(self.surf, (0, 0))
		if self.render_mode == "human":
			pygame.event.pump()
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()

		elif self.render_mode == "rgb_array":
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
			)
