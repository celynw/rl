"""
Based on CartPole-v1 from gym=0.23.1
Changed colours to increase contrast.

Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from multiprocessing import Event
from pathlib import Path
import time
from typing import Optional, Union

import numpy as np
import pygame
from pygame import gfxdraw
import cv2
import torch

import gym
from gym import spaces, logger
from gym.envs.classic_control.cartpole import CartPoleEnv

# ==================================================================================================
# class CartPoleEnvRGB(gym.Env[np.ndarray, Union[int, np.ndarray]]):
class CartPoleEnvRGB(gym.Env):
	"""
	### Description

	This environment corresponds to the version of the cart-pole problem
	described by Barto, Sutton, and Anderson in ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
	A pole is attached by an un-actuated joint to a cart, which moves along a
	frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

	### Action Space

	The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

	| Num | Action                 |
	|-----|------------------------|
	| 0   | Push cart to the left  |
	| 1   | Push cart to the right |

	**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

	### Observation Space

	The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

	| Num | Observation           | Min                  | Max                |
	|-----|-----------------------|----------------------|--------------------|
	| 0   | Cart Position         | -4.8                 | 4.8                |
	| 1   | Cart Velocity         | -Inf                 | Inf                |
	| 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
	| 3   | Pole Angular Velocity | -Inf                 | Inf                |

	**Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
	-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
	-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

	### Rewards

	Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken, including the termination step, is allotted. The threshold for rewards is 475 for v1.

	### Starting State

	All observations are assigned a uniformly random value in `(-0.05, 0.05)`

	### Episode Termination

	The episode terminates if any one of the following occurs:
	1. Pole Angle is greater than ±12°
	2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
	3. Episode length is greater than 500 (200 for v0)

	### Arguments

	```
	gym.make('CartPole-v1')
	```

	No additional arguments are currently supported.
	"""

	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = self.masspole + self.masscart
		self.length = 0.5 # actually half the pole's length
		self.polemass_length = self.masspole * self.length
		self.force_mag = 10.0
		self.tau = 0.02 # seconds between state updates
		self.kinematics_integrator = "euler"

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# Angle limit set to 2 * theta_threshold_radians so failing observation
		# is still within bounds.
		high = np.array(
			[
				self.x_threshold * 2,
				np.finfo(np.float32).max,
				self.theta_threshold_radians * 2,
				np.finfo(np.float32).max,
			],
			dtype=np.float32,
		)

		self.screen_width = 600
		# self.screen_height = 200
		self.screen_height = 400
		# self.screen_width_ = 90
		# self.screen_width_ = 600
		self.screen_width_ = 240
		# self.screen_height_ = 40
		# self.screen_height_ = 400
		self.screen_height_ = 64

		self.action_space = spaces.Discrete(2)
		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		# self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.screen_height, self.screen_width, 2))
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 2)) # Accumulate
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 4)) # EV-FlowNet
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height_, self.screen_width_, 4)) # EV-FlowNet
		self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(3, self.screen_height_, self.screen_width_)) # EDeNN
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(3, self.screen_height_, self.screen_width_)) # EDeNN # DEBUG

		self.screen = None
		self.clock = None
		self.isopen = True
		self.state = None
		self.connected = False
		self.updatedPolicy = False

		self.steps_beyond_done = None

		# DEBUG
		# print("WARNING, setting env seed")

		self.model = None

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb):
		# Crop
		# print(rgb.shape)
		rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# print(rgb.shape)
		# Resize
		rgb = cv2.resize(rgb, (self.screen_width_, self.screen_height_), interpolation=cv2.INTER_AREA)
		# cv2.imwrite("cartpole_resized.png", rgb)
		# cv2.imwrite(f"/code/toys/debug_training/dataset/cartpole_resized.png", rgb)
		# quit(0) # DEBUG

		return rgb

	# ----------------------------------------------------------------------------------------------
	def set_state(self, state):
		self.state = state

	# ----------------------------------------------------------------------------------------------
	def set_updatedPolicy(self):
		self.updatedPolicy = True

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		self.info = {
			"failReason": None,
			"updatedPolicy": int(self.updatedPolicy),
			"state": np.zeros(4),
		}

		# print(f"action.shape: {action.shape}")
		# print(f"action: {action}")
		# quit(0)
		err_msg = f"{action!r} ({type(action)}) invalid"
		assert self.action_space.contains(action), err_msg
		assert self.state is not None, "Call reset before using step method."
		x, x_dot, theta, theta_dot = self.state
		force = self.force_mag if action == 1 else -self.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)

		# For the interested reader:
		# https://coneural.org/florian/papers/05_cart_pole.pdf
		temp = (
			force + self.polemass_length * theta_dot ** 2 * sintheta
		) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta * temp) / (
			self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
		)
		xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

		if self.kinematics_integrator == "euler":
			x = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else: # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot

		self.state = (x, x_dot, theta, theta_dot)

		done = bool(
			x < -self.x_threshold
			or x > self.x_threshold
			or theta < -self.theta_threshold_radians
			or theta > self.theta_threshold_radians
		)

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn(
					"You are calling 'step()' even though this "
					"environment has already returned done = True. You "
					"should always call 'reset()' once you receive 'done = "
					"True' -- any further steps are undefined behavior."
				)
			self.steps_beyond_done += 1
			reward = 0.0

		# print("render")
		rgb = self.render("rgb_array")
		rgb = self.resize(rgb)
		rgb = np.transpose(rgb, (2, 0, 1)) # HWC to CHW

		if x < -self.x_threshold:
			self.info["failReason"] = "too_far_left"
		elif x > self.x_threshold:
			self.info["failReason"] = "too_far_right"
		elif theta < -self.theta_threshold_radians:
			self.info["failReason"] = "pole_fell_left"
		elif theta > self.theta_threshold_radians:
			self.info["failReason"] = "pole_fell_right"

		if done:
			# We're not doing setting this to False immediately because the monitor only writes a line when an episode is done
			self.updatedPolicy = False

		self.info["state"] = self.state

		return rgb, reward, done, self.info

	# ----------------------------------------------------------------------------------------------
	def reset(
		self,
		*,
		seed: Optional[int] = None,
		return_info: bool = False,
		options: Optional[dict] = None,
	):
		# DEBUG
		# env_.seed(0)
		# seed = 0

		super().reset(seed=seed)
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = None
		# print("render")

		if self.model is not None:
			self.model.reset_env()
		else:
			print("WARNING: env model is None")

		if not return_info:
			return torch.zeros(3, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
		else:
			return torch.zeros(3, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), {}

	# ----------------------------------------------------------------------------------------------
	def render(self, mode="human"):
		world_width = self.x_threshold * 2
		scale = self.screen_width / world_width
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.state is None:
			return None

		x = self.state

		if self.screen is None:
			pygame.init()
			pygame.display.init()
			self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		if self.clock is None:
			self.clock = pygame.time.Clock()

		self.surf = pygame.Surface((self.screen_width, self.screen_height))
		self.surf.fill((255, 255, 255))

		l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
		axleoffset = cartheight / 4.0
		cartx = x[0] * scale + self.screen_width / 2.0 # MIDDLE OF CART
		carty = 100 # TOP OF CART
		# carty = cartheight # TOP OF CART
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

		if mode == "rgb_array":
			arr = np.array(pygame.surfarray.pixels3d(self.screen))
			# arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) # FIX maybe avoid using cv2 here and use np instead?
			return np.transpose(arr, axes=(1, 0, 2))
		else:
			return self.isopen

	# ----------------------------------------------------------------------------------------------
	def close(self):
		if self.screen is not None:
			pygame.display.quit()
			pygame.quit()
			self.isopen = False

	# ----------------------------------------------------------------------------------------------
	def set_model(self, model):
		self.model = model
