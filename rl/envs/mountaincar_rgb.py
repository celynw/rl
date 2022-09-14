"""
Based on MountainCar-v0 from gym=0.25.1
"""
import math
from typing import Optional

import numpy as np
import torch
from gym import spaces
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.error import DependencyNotInstalled
import cv2

# ==================================================================================================
class MountainCarEnvRGB(MountainCarEnv):
	state_shape = (2, ) # TODO unused for now
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		self.screen_width_ = 150
		self.screen_height_ = 100
		MountainCarEnv.__init__(self, render_mode=None)
		self.screen_width = 150
		self.screen_height = 100
		self.updatedPolicy = False
		self.model = None

		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		# self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(3, self.screen_height_, self.screen_width_))
		self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(1, self.screen_height_, self.screen_width_))
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(3, self.screen_height_, self.screen_width_))
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(1, self.screen_height_, self.screen_width_))

	# ----------------------------------------------------------------------------------------------
	def set_model(self, model):
		self.model = model

	# ----------------------------------------------------------------------------------------------
	def set_state(self, state):
		self.state = state

	# ----------------------------------------------------------------------------------------------
	def set_updatedPolicy(self):
		self.updatedPolicy = True

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb):
		# Crop
		# rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# Resize
		# rgb = cv2.resize(rgb, (self.screen_width_, self.screen_height_), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def _render(self, mode="human"):
		assert mode in self.metadata["render_modes"]
		try:
			import pygame
			from pygame import gfxdraw
		except ImportError:
			raise DependencyNotInstalled(
				"pygame is not installed, run `pip install gym[classic_control]`"
			)

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

		world_width = self.max_position - self.min_position
		scale = self.screen_width / world_width
		carwidth = 20
		carheight = 10

		self.surf = pygame.Surface((self.screen_width, self.screen_height))
		self.surf.fill((255, 255, 255))

		pos = self.state[0]

		xs = np.linspace(self.min_position, self.max_position, 100)
		ys = self._height(xs)
		xys = list(zip((xs - self.min_position) * scale, ys * scale))

		pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

		clearance = 3

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

		gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
		gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

		for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
			c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
			wheel = (
				int(c[0] + (pos - self.min_position) * scale),
				int(c[1] + clearance + self._height(pos) * scale),
			)

			# gfxdraw.aacircle(
			# 	self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
			# )
			# gfxdraw.filled_circle(
			# 	self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
			# )

		flagx = int((self.goal_position - self.min_position) * scale)
		flagy1 = int(self._height(self.goal_position) * scale)
		flagy2 = flagy1 + 50
		gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

		gfxdraw.aapolygon(
			self.surf,
			[(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
			(204, 204, 0),
		)
		gfxdraw.filled_polygon(
			self.surf,
			[(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
			(204, 204, 0),
		)

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

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		_, reward, terminated, truncated, _ = super().step(action)
		info = {
			"updatedPolicy": int(self.updatedPolicy),
			"state": self.state,
		}

		rgb = self.render("rgb_array")
		rgb = self.resize(rgb)
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
		rgb = rgb[None, ...] # Not sure why I need to do this here but not for reset() or in CartPoleEnvRGB...

		if terminated:
			# We're not doing setting this to False immediately because the monitor only writes a line when an episode is done
			self.updatedPolicy = False

		return rgb, reward, terminated, truncated, info

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
		rgb = self.render("rgb_array")
		rgb = self.resize(rgb)
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
		if self.model is not None:
			self.model.reset_env()
		else:
			print("WARNING: env model is None")

		if not return_info:
			# return torch.zeros(3, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
			return torch.zeros(1, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
		else:
			# return torch.zeros(3, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), info
			return torch.zeros(1, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), info
