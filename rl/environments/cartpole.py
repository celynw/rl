import argparse
from typing import Optional

import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
import numpy as np
import cv2
import pygame, pygame.gfxdraw

from rl.environments.utils import EventEnv

# ==================================================================================================
class CartPoleEvents(EventEnv, CartPoleEnv):
	# ----------------------------------------------------------------------------------------------
	# def __init__(self, args: argparse.Namespace, event_image: bool = False, return_rgb: bool = False, output_width: int = 126, output_height: int = 84): # if not cropping
	def __init__(self, args: argparse.Namespace, event_image: bool = False, return_rgb: bool = False, output_width: int = 240, output_height: int = 64): # if cropping
		"""
		Event version of CartPole environment.

		Args:
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
			event_image (bool, optional): Accuumlates events into an event image. Defaults to False.
		"""
		self.output_width = output_width
		self.output_height = output_height
		self.updatedPolicy = False # Used for logging whenever the policy is updated
		self.render_events = False
		CartPoleEnv.__init__(self, render_mode="rgb_array")
		EventEnv.__init__(self, self.output_width, self.output_height, args, event_image) # type: ignore
		# self.iter = 0
		self.failReason = None

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
			tuple[np.ndarray, float, bool, bool]: Step returns.
		"""
		_, reward, terminated, truncated, _ = super().step(action) # type: ignore
		events = self.observe()

		x, _, theta, _ = self.state
		if x < -self.x_threshold:
			self.failReason = "too_far_left"
		elif x > self.x_threshold:
			self.failReason = "too_far_right"
		elif theta < -self.theta_threshold_radians:
			self.failReason = "pole_fell_left"
		elif theta > self.theta_threshold_radians:
			self.failReason = "pole_fell_right"
		else:
			self.failReason = None

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		# self.iter += 1
		# events = np.ones_like(events.numpy()) * self.iter

		return events.numpy(), reward, terminated, truncated, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict:
		"""
		Return a created dictionary for the step info.

		Returns:
			dict: Key-value pairs for the step info.
		"""
		info = EventEnv.get_info(self)
		info["failReason"] = self.failReason

		return info

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
		rgb = cv2.resize(rgb, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def render(self) -> np.ndarray:
		"""
		Copied and adapted from CartPoleEnv.render() [gym==0.26.2].

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
		pygame.gfxdraw.aapolygon(self.surf, pole_coords, (0, 0, 0)) # NOTE: Making the pole black
		pygame.gfxdraw.filled_polygon(self.surf, pole_coords, (0, 0, 0)) # NOTE: Making the pole black

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


# ==================================================================================================
class CartPoleRGB(CartPoleEnv):
	# ----------------------------------------------------------------------------------------------
	# def __init__(self, args: argparse.Namespace, output_width: int = 126, output_height: int = 84): # if not cropping
	def __init__(self, args: argparse.Namespace, output_width: int = 240, output_height: int = 64): # if cropping
		"""
		RGB version of CartPole environment.

		Args:
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
		"""
		self.output_width = output_width
		self.output_height = output_height
		self.updatedPolicy = False # Used for logging whenever the policy is updated
		super().__init__(render_mode="rgb_array")
		self.state_space = self.observation_space # For access later
		# FIX: I should normalise my observation space (well, both), but not sure how to for event tensor
		self.shape = [3, self.output_height, self.output_width]
		self.observation_space = spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)
		# self.iter = 0
		self.failReason = None

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
	def set_updatedPolicy(self):
		"""
		Set a boolean flag. Used for logging whenever the policy is updated.
		TODO move to model. It's only in the env for program accessibility...
		"""
		self.updatedPolicy = True

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
		rgb = self.render()
		rgb = self.resize(rgb)
		rgb = np.transpose(rgb, (2, 0, 1)) # HWC -> CHW

		x, _, theta, _ = self.state
		if x < -self.x_threshold:
			self.failReason = "too_far_left"
		elif x > self.x_threshold:
			self.failReason = "too_far_right"
		elif theta < -self.theta_threshold_radians:
			self.failReason = "pole_fell_left"
		elif theta > self.theta_threshold_radians:
			self.failReason = "pole_fell_right"
		else:
			self.failReason = None

		if terminated: # Monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return rgb, reward, terminated, truncated, self.get_info()

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict:
		"""
		Return a created dictionary for the step info.

		Returns:
			dict: Key-value pairs for the step info.
		"""
		info = EventEnv.get_info(self)
		info["failReason"] = self.failReason

		return info

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
		"""
		Resets the environment.

		Args:
			seed (int, optional): The seed that is used to initialize the environment's PRNG. Defaults to None.
			options (dict, optional): Additional information to specify how the environment is reset. Defaults to None.

		Returns:
			np.ndarray: First observation.
		"""
		super().reset(seed=seed, options=options) # NOTE: Not using the output

		rgb = self.render()
		rgb = self.resize(rgb)
		rgb = np.transpose(rgb, (2, 0, 1)) # HWC -> CHW

		return rgb, self.get_info()

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
		rgb = cv2.resize(rgb, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)

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
			elif self.render_mode == "rgb_array":
				self.screen = pygame.Surface((self.screen_width, self.screen_height))
			else:
				raise ValueError(f"Invalid render mode? 'self.render_mode'")
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
		pygame.gfxdraw.aapolygon(self.surf, pole_coords, (0, 0, 0)) # NOTE: Making the pole black
		pygame.gfxdraw.filled_polygon(self.surf, pole_coords, (0, 0, 0)) # NOTE: Making the pole black

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
