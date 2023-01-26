import argparse

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.error import DependencyNotInstalled
import numpy as np
import cv2

from rl.environments.utils import EventEnv

# ==================================================================================================
class PendulumEvents(EventEnv, PendulumEnv):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, event_image: bool = False, output_width: int = 64, output_height: int = 64): # if cropping
		"""
		Event version of CartPole environment.
		"""
		self.output_width = output_width
		self.output_height = output_height
		self.updatedPolicy = False # Used for logging whenever the policy is updated
		self.render_events = False
		PendulumEnv.__init__(self, render_mode="rgb_array")
		# self.state_space = self.observation_space
		EventEnv.__init__(self, self.output_width, self.output_height, args, event_image) # type: ignore
		self.state_space = spaces.Box(low=-16.2736044, high=0, shape=(2,), dtype=np.float32) # FIX It's wrong in EventEnv I think, check other environments
		# self.iter = 0

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = EventEnv.add_argparse_args(parser)
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L17
		# parser.set_defaults(steps=1e5)
		parser.set_defaults(n_envs=4)
		parser.set_defaults(n_steps=1024)
		parser.set_defaults(gae_lambda=0.95)
		parser.set_defaults(gamma=0.9)
		parser.set_defaults(n_epochs=10)
		parser.set_defaults(ent_coef=0.0)
		parser.set_defaults(lr=1e-3)
		parser.set_defaults(clip_range=0.2)
		parser.set_defaults(batch_size=256)

		return parser

	# ----------------------------------------------------------------------------------------------
	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
		"""
		Perform a single environment step.

		Args:
			action (int): Which action to perform this step.

		Returns:
			tuple[np.ndarray, float, bool, bool]: Step returns.
		"""
		_, reward, terminated, truncated, _ = super().step(action) # type: ignore
		events = self.observe()

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		# self.iter += 1
		# events = np.ones_like(events.numpy()) * self.iter

		return events.numpy(), reward, terminated, truncated, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb: np.ndarray) -> np.ndarray:
		"""
		Crop the top and bottom, then reduce the resolution.

		Args:
			rgb (np.ndarray): OpenCV render of the observation.

		Returns:
			np.ndarray: Resized image.
		"""
		# Crop
		rgb = rgb[122:122 + 256, 122:122 + 256, :]
		# Resize
		rgb = cv2.resize(rgb, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def render(self):
		if self.render_mode is None:
			assert self.spec is not None
			gym.logger.warn(
				"You are calling render method without specifying any render mode. "
				"You can specify the render_mode at initialization, "
				f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
			)
			return

		try:
			import pygame
			from pygame import gfxdraw
		except ImportError as e:
			raise DependencyNotInstalled(
				"pygame is not installed, run `pip install gymnasium[classic_control]`"
			) from e

		if self.screen is None:
			pygame.init()
			if self.render_mode == "human":
				pygame.display.init()
				self.screen = pygame.display.set_mode(
					(self.screen_dim, self.screen_dim)
				)
			else: # mode in "rgb_array"
				self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
		if self.clock is None:
			self.clock = pygame.time.Clock()

		self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
		self.surf.fill((255, 255, 255))

		bound = 2.2
		scale = self.screen_dim / (bound * 2)
		offset = self.screen_dim // 2

		rod_length = 1 * scale
		rod_width = 0.2 * scale
		l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
		coords = [(l, b), (l, t), (r, t), (r, b)]
		transformed_coords = []
		for c in coords:
			c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
			c = (c[0] + offset, c[1] + offset)
			transformed_coords.append(c)
		gfxdraw.aapolygon(self.surf, transformed_coords, (0, 0, 0))
		gfxdraw.filled_polygon(self.surf, transformed_coords, (0, 0, 0))

		gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (0, 0, 0))
		gfxdraw.filled_circle(
			self.surf, offset, offset, int(rod_width / 2), (0, 0, 0)
		)

		rod_end = (rod_length, 0)
		rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
		rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
		gfxdraw.aacircle(
			self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (0, 0, 0)
		)
		gfxdraw.filled_circle(
			self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (0, 0, 0)
		)

		# fname = path.join(path.dirname(__file__), "assets/clockwise.png")
		# img = pygame.image.load(fname)
		# if self.last_u is not None:
		# 	scale_img = pygame.transform.smoothscale(
		# 		img,
		# 		(scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
		# 	)
		# 	is_flip = bool(self.last_u > 0)
		# 	scale_img = pygame.transform.flip(scale_img, is_flip, True)
		# 	self.surf.blit(
		# 		scale_img,
		# 		(
		# 			offset - scale_img.get_rect().centerx,
		# 			offset - scale_img.get_rect().centery,
		# 		),
		# 	)

		# drawing axle
		gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
		gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

		self.surf = pygame.transform.flip(self.surf, False, True)
		self.screen.blit(self.surf, (0, 0))
		if self.render_mode == "human":
			pygame.event.pump()
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()

		else: # mode == "rgb_array":
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
			)
