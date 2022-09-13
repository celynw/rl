"""
Based on CartPole-v1 from gym=0.25.1
Changed colours to increase contrast.
"""
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw
import cv2
import torch
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv

from rl.envs.utils import EventEnv

# ==================================================================================================
class CartPoleEnvEvents(EventEnv, CartPoleEnv):
	state_shape = (4, ) # TODO unused for now
	# ----------------------------------------------------------------------------------------------
	def __init__(self, init_ros: bool = True, tsamples: int = 10, event_image: bool = False):
		self.screen_width_ = 240
		self.screen_height_ = 64
		EventEnv.__init__(self, self.screen_width_, self.screen_height_, init_ros, tsamples, event_image)
		CartPoleEnv.__init__(self, render_mode=None)

		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		# self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.screen_height, self.screen_width, 2))
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 2)) # Accumulate
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 4)) # EV-FlowNet
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height_, self.screen_width_, 4)) # EV-FlowNet
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))
		if self.event_image:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
		else:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb):
		# Crop
		rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# Resize
		rgb = cv2.resize(rgb, (self.screen_width_, self.screen_height_), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		_, reward, terminated, truncated, _ = super().step(action)
		info = {
			"failReason": None,
			"updatedPolicy": int(self.updatedPolicy),
			"state": self.state,
		}

		event_tensor = self.get_events()
		if self.event_image:
			event_tensor = event_tensor.sum(1)
			# event_tensor = event_tensor.bool().double() * 255

		x, _, theta, _ = self.state
		if x < -self.x_threshold:
			info["failReason"] = "too_far_left"
		elif x > self.x_threshold:
			info["failReason"] = "too_far_right"
		elif theta < -self.theta_threshold_radians:
			info["failReason"] = "pole_fell_left"
		elif theta > self.theta_threshold_radians:
			info["failReason"] = "pole_fell_right"

		if terminated:
			# We're not doing setting this to False immediately because the monitor only writes a line when an episode is terminated
			self.updatedPolicy = False

		return event_tensor.numpy(), reward, terminated, truncated, info

	# ----------------------------------------------------------------------------------------------
	def reset(
		self,
		*,
		seed: Optional[int] = None,
		return_info: bool = False,
		options: Optional[dict] = None,
	):
		# Not using the output of super().reset()
		super().reset(seed=seed, return_info=return_info, options=options)
		info = {"state": self.state}

		# Initialise ESIM, need two frames to get a difference to generate events
		self.get_events(wait=False)
		event_tensor = self.get_events()
		if self.event_image:
			event_tensor = event_tensor.sum(1)
			# event_tensor = event_tensor.bool().double() * 255 # DEBUG
		if self.model is not None:
			self.model.reset_env()
		else:
			print("WARNING: env model is None")

		if not return_info:
			if self.event_image:
				return torch.zeros(2, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
			else:
				return torch.zeros(2, self.tsamples, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
		else:
			if self.event_image:
				return torch.zeros(2, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), info
			else:
				return torch.zeros(2, self.tsamples, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), info

	# ----------------------------------------------------------------------------------------------
	def render(self, mode="human"):
		# Copied and adapted from super.render()
		assert mode in self.metadata["render_modes"]
		if self.screen is None:
			pygame.init()
			if mode == "human":
				pygame.display.init()
				self.screen = pygame.display.set_mode(
					(self.screen_width, self.screen_height)
				)
			else: # mode in {"rgb_array", "single_rgb_array"}
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
		gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 50))
		gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 50))

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
		gfxdraw.aapolygon(self.surf, pole_coords, (50, 0, 0))
		gfxdraw.filled_polygon(self.surf, pole_coords, (50, 0, 0))

		# Axle
		# gfxdraw.aacircle(
		# 	self.surf,
		# 	int(cartx),
		# 	int(carty + axleoffset),
		# 	int(polewidth / 2),
		# 	(129, 132, 203),
		# )
		# gfxdraw.filled_circle(
		# 	self.surf,
		# 	int(cartx),
		# 	int(carty + axleoffset),
		# 	int(polewidth / 2),
		# 	(129, 132, 203),
		# )

		# Ground
		# gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

		self.surf = pygame.transform.flip(self.surf, False, True)
		self.screen.blit(self.surf, (0, 0))
		if mode == "human":
			pygame.event.pump()
			self.clock.tick(self.metadata["render_fps"])
			pygame.display.flip()
		elif mode in {"rgb_array", "single_rgb_array"}:
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
			)
