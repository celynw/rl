import argparse
from typing import Optional, Union

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import spaces
import numpy as np
import cv2
import pygame, pygame.gfxdraw
import gym

from rl.environments.utils import EventEnv
from rl.models import EDeNN, SNN

# ==================================================================================================
class CartPoleEvents(EventEnv, CartPoleEnv):
	state_space: spaces.Space
	model: Optional[Union[EDeNN, SNN]] = None
	# ----------------------------------------------------------------------------------------------
	def __init__(self, fps: Optional[int] = None, tsamples: int = 10, event_image: bool = False,
			return_rgb: bool = False):
		"""
		Event version of CartPole environment

		Args:
			fps (int, optional): Frames per second for event simulation and step times. Defaults to the environment's default.
			tsamples (int, optional): Number of time bins in the observations. Defaults to 10.
			event_image (bool, optional): Accuumlates events into an event image. Defaults to False.
			return_rgb (bool, optional): _description_. Defaults to False.
		"""
		self.return_rgb = return_rgb
		self.events_width = 240
		self.events_height = 64
		CartPoleEnv.__init__(self, render_mode="rgb_array")
		EventEnv.__init__(self, self.events_width, self.events_height, fps, tsamples, event_image) # type: ignore

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		parser = EventEnv.add_argparse_args(parser)
		# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L32
		# parser.set_defaults(steps=1e5)
		parser.set_defaults(n_steps=32 * 8) # n_envs = 8, rollout buffer size is n_steps * n_envs
		parser.set_defaults(gae_lambda=0.8)
		parser.set_defaults(gamma=0.98)
		parser.set_defaults(n_epochs=20)
		parser.set_defaults(ent_coef=0.0)
		parser.set_defaults(lr=0.001)
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
			tuple[np.ndarray, float, bool, bool, dict]: Step returns.
		"""
		_, reward, terminated, truncated, _ = super().step(action) # type: ignore
		events = self.observe()

		info = super().get_info()
		info["failReason"] = None
		x, _, theta, _ = self.state
		if x < -self.x_threshold:
			info["failReason"] = "too_far_left"
		elif x > self.x_threshold:
			info["failReason"] = "too_far_right"
		elif theta < -self.theta_threshold_radians:
			info["failReason"] = "pole_fell_left"
		elif theta > self.theta_threshold_radians:
			info["failReason"] = "pole_fell_right"

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return events.numpy(), reward, terminated, truncated, info

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
		rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# Resize
		rgb = cv2.resize(rgb, (self.events_width, self.events_height), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def render(self) -> np.ndarray:
		"""
		Copied and adapted from CartPoleEnv.render() [gym==0.26.2].

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
			else: # self.render_mode == "rgb_array"
				self.screen = pygame.Surface((self.screen_width, self.screen_height))
		if self.clock is None:
			self.clock = pygame.time.Clock()

		world_width = self.x_threshold * 2
		scale = self.screen_width / world_width
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.state is None:
			return None

		x = self.state

		self.surf = pygame.Surface((self.screen_width, self.screen_height))
		self.surf.fill((255, 255, 255))

		l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
		axleoffset = cartheight / 4.0
		cartx = x[0] * scale + self.screen_width / 2.0 # MIDDLE OF CART
		carty = 100 # TOP OF CART
		cart_coords = [(l, b), (l, t), (r, t), (r, b)]
		cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
		pygame.gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
		pygame.gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

		l, r, t, b = (
			-polewidth / 2,
			polewidth / 2,
			polelen - polewidth / 2,
			-polewidth / 2,
		)

		pole_coords = []
		for coord in [(l, b), (l, t), (r, t), (r, b)]:
			coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
			coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
			pole_coords.append(coord)
		pygame.gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
		pygame.gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

		# NOTE: Not drawing the axle or ground
		# pygame.gfxdraw.aacircle(
		# 	self.surf,
		# 	int(cartx),
		# 	int(carty + axleoffset),
		# 	int(polewidth / 2),
		# 	(129, 132, 203),
		# )
		# pygame.gfxdraw.filled_circle(
		# 	self.surf,
		# 	int(cartx),
		# 	int(carty + axleoffset),
		# 	int(polewidth / 2),
		# 	(129, 132, 203),
		# )

		# pygame.gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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
