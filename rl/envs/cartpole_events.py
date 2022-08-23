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

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray, Event

import gym
from gym import spaces, logger
# from gym.envs.classic_control import cartpole

# use_4D = True # Use 4D tensor instead of 5D tensor, for debug training only
use_4D = False

# ==================================================================================================
# class CartPoleEnvEvents(gym.Env[np.ndarray, Union[int, np.ndarray]]):
class CartPoleEnvEvents(gym.Env):
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
	def __init__(self, init_ros: bool = True):
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

		# self.tsamples = 100 # TODO
		# self.tsamples = 20 # TODO
		self.tsamples = 10 # TODO
		# self.tsamples = 1 # TODO

		self.init_ros = init_ros

		self.action_space = spaces.Discrete(2)
		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		# self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		# self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.screen_height, self.screen_width, 2))
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 2)) # Accumulate
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height, self.screen_width, 4)) # EV-FlowNet
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(self.screen_height_, self.screen_width_, 4)) # EV-FlowNet
		# self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))
		if use_4D:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_)) # EDeNN
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_)) # EDeNN # DEBUG
		else:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_)) # EDeNN
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_)) # EDeNN # DEBUG

		self.screen = None
		self.clock = None
		self.isopen = True
		self.state = None
		self.physics_state = None
		self.connected = False
		self.updatedPolicy = False

		self.steps_beyond_done = None

		# ROS
		name = Path(__file__).stem# if name is None else None
		debug = False
		if self.init_ros:
			rospy.init_node(name, anonymous=False, log_level=rospy.DEBUG if debug else rospy.INFO)
		self.bridge = CvBridge()
		self.pub_image = rospy.Publisher("image", Image, queue_size=10)
		self.sub_events = rospy.Subscriber("/cam0/events", EventArray, self.callback)
		# self.events_msg = None
		self.events = None
		if self.init_ros:
			self.time = rospy.Time.now()
		# self.generator = SpikeRepresentationGenerator(self.screen_height, self.screen_width, self.tsamples)
		self.generator = SpikeRepresentationGenerator(self.screen_height_, self.screen_width_, self.tsamples)

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
		self.physics_state = state

	# ----------------------------------------------------------------------------------------------
	def set_updatedPolicy(self):
		self.updatedPolicy = True

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		self.info = {
			"failReason": None,
			"updatedPolicy": int(self.updatedPolicy),
			"physicsState": np.zeros(4),
		}

		# print(f"action.shape: {action.shape}")
		# print(f"action: {action}")
		# quit(0)
		err_msg = f"{action!r} ({type(action)}) invalid"
		assert self.action_space.contains(action), err_msg
		assert self.physics_state is not None, "Call reset before using step method."
		x, x_dot, theta, theta_dot = self.physics_state
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

		self.physics_state = (x, x_dot, theta, theta_dot)

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
		# print("publish")
		if self.init_ros:
			self.publish(rgb)

		# print("events")
		events = self.eventify()

		# event_tensor = self.tensorify_accumulate(events)
		# # DEBUG sum all bins, permute for gym compatibility
		# event_tensor = event_tensor.sum(dim=1)

		# event_tensor = self.tensorify_evflownet(events)

		# print("tensor")
		event_tensor = self.tensorify_edenn(events)
		if use_4D:
			event_tensor = event_tensor.sum(1) # DEBUG
			# # DEBUG AGAIN
			# event_tensor = event_tensor.bool().double() * 255
		# print(f"event_tensor.shape: {event_tensor.shape}")

		# # event_tensor = event_tensor.permute(1, 2, 0)
		# event_tensor = event_tensor.permute(1, 2, 3, 0)
		# print(f"event_tensor.shape: {event_tensor.shape}")
		# # print(event_tensor.unique(return_counts=True))

		# return event_tensor, reward, done, {}
		# print(f"step returning: {event_tensor.numpy().shape}")

		# print("return")

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

		self.info["physicsState"] = self.physics_state

		return event_tensor.numpy(), reward, done, self.info

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
		self.physics_state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = None
		# print("render")
		rgb = self.render("rgb_array")

		rgb = self.resize(rgb)
		# print("publish")
		if self.init_ros:
			self.publish(rgb)

		if self.model is not None:
			self.model.reset_env()
		else:
			print("WARNING: env model is None")

		if not return_info:
			# return EventArray()
			# return torch.tensor(0)
			# return torch.zeros(self.screen_height, self.screen_width, 2, dtype=torch.double) # Accumulate
			# return torch.zeros(self.screen_height, self.screen_width, 4, dtype=torch.double) # EV-FlowNet
			# return torch.zeros(self.screen_height_, self.screen_width_, 4, dtype=torch.double) # EV-FlowNet
			# return torch.zeros(self.screen_height_, self.screen_width_, 4, dtype=torch.double).numpy() # EV-FlowNet
			if use_4D:
				return torch.zeros(2, self.screen_height_, self.screen_width_, dtype=torch.double).numpy()
			else:
				return torch.zeros(2, self.tsamples, self.screen_height_, self.screen_width_, dtype=torch.double).numpy() # EDeNN
		else:
			# return EventArray(), {}
			# return torch.tensor(0), {}
			# return torch.zeros(self.screen_height, self.screen_width, 2, dtype=torch.double), {} # Accumulate
			# return torch.zeros(self.screen_height, self.screen_width, 4, dtype=torch.double), {} # EV-FlowNet
			# return torch.zeros(self.screen_height_, self.screen_width_, 4, dtype=torch.double), {} # EV-FlowNet
			# return torch.zeros(self.screen_height_, self.screen_width_, 4, dtype=torch.double).numpy(), {} # EV-FlowNet
			if use_4D:
				return torch.zeros(2, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), {}
			else:
				return torch.zeros(2, self.tsamples, self.screen_height_, self.screen_width_, dtype=torch.double).numpy(), {} # EDeNN

	# ----------------------------------------------------------------------------------------------
	def publish(self, obs):
		try:
			msg = self.bridge.cv2_to_imgmsg(obs, encoding="rgb8")
		except CvBridgeError as e:
			print(e)
		# msg.header.frame_id = self.params.cam_frame
		msg.header.frame_id = "cam"
		# print(f"Publishing for {self.time}")
		msg.header.stamp = self.time
		# print(f"Publishing seq {msg.header.seq}")
		# self.events_msg = None
		self.events = None
		self.pub_image.publish(msg)

		i = 0
		while not self.connected:
			print(f"Waiting for subscriber on '{self.pub_image.name}' topic ({i})")
			# Check that ESIM is active
			# FIX this doesn't account for _what_ is listening, it could be just a visualiser or rosbag!
			if self.pub_image.get_num_connections() > 0:
				self.connected = True
			i += 1
			time.sleep(1)

		self.time += rospy.Duration(1 / 30)

		# DEBUG
		# cv2.imwrite(f"/tmp/events_{self.time.to_sec()}.png", obs)

	# ----------------------------------------------------------------------------------------------
	def eventify(self):
		# events_msg = EventArray()
		# try:
		# 	events_msg = rospy.wait_for_message("/cam0/events", EventArray, timeout=1.0)
		# 	# events_msg = rospy.wait_for_message("/cam0/events", EventArray, timeout=None)
		# except:
		# 	print("Timeout")
		# 	pass

		# while self.events_msg is None:
		# while len(self.events) == 0:
		while self.events is None:
			pass

		if rospy.is_shutdown():
			print("Node is shutting down")
			quit(0)
		# events_msg = self.events_msg
		# self.events_msg = None

		# return events_msg
		# return self.events_msg
		return self.events

	# ----------------------------------------------------------------------------------------------
	# def tensorify_accumulate(self, msg: EventArray):
	def tensorify_accumulate(self, events: list[Event]):
		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		events = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		events = events.permute(0, 3, 1, 2) # CHWT -> CDHW

		return events

	# ----------------------------------------------------------------------------------------------
	def tensorify_evflownet(self, events: list[Event]):
		# 4-channels * H * W:
		# - 0: number of positive
		# - 1: number of negative
		# - 2: most recent timestamp of positive
		# - 3: most recent timestamp of negative
		# I'm assuming the events list is sorted! By timestamp ascending

		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		spike_tensor = torch.zeros((4, self.screen_height_, self.screen_width_))
		if len(stamps) == 0:
			return spike_tensor # Empty tensor, don't raise an error
		# spike_tensor[polarities.long(), coords[:, 1].long(), coords[:, 0].long()] += 1
		# # But what about the timestamps?
		# FIX better to use start time which is the start time of this frame, not first event!
		first_timestamp = stamps[0]
		for event in events:
			# Increment number of events
			spike_tensor[int(event.polarity), event.y, event.x] += 1
			# Update timestamp
			ts = event.ts.to_sec() - first_timestamp
			if ts > spike_tensor[int(event.polarity) + 2, event.y, event.x]:
				spike_tensor[int(event.polarity) + 2, event.y, event.x] = ts

		return spike_tensor

	# ----------------------------------------------------------------------------------------------
	def tensorify_edenn(self, events: list[Event]):
		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		events = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		events = events.permute(0, 3, 1, 2) # CHWT -> CDHW

		return events

	# ----------------------------------------------------------------------------------------------
	def callback(self, msg):
		# self.events_msg = msg
		# self.events += msg.events
		self.events = msg.events

	# ----------------------------------------------------------------------------------------------
	def render(self, mode="human"):
		world_width = self.x_threshold * 2
		scale = self.screen_width / world_width
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.physics_state is None:
			return None

		x = self.physics_state

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


# ==================================================================================================
class SpikeRepresentationGenerator:
	"""Generate spikes from event tensors."""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, height: int, width: int, num_time_bins: int):
		"""
		Generate spikes from event tensors.
		Args:
			height (int): Height of event image
			width (int): Width of event image
		"""
		self.height = height
		self.width = width
		self.num_time_bins = num_time_bins

	# ----------------------------------------------------------------------------------------------
	def getSlayerSpikeTensor(self, ev_pol: torch.Tensor, ev_xy: torch.Tensor,
				ev_ts_us: torch.Tensor) -> torch.Tensor:
		"""
		Generate spikes from event tensors.
		All arguments must be of the same image shape.
		Args:
			ev_pol (torch.Tensor): Event polarities
			ev_xy (torch.Tensor): Event locations
			ev_ts_us (torch.Tensor): Event timestamps in microseconds
		Returns:
			torch.Tensor: Spike train tensor
		"""
		spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))
		if len(ev_ts_us) < 2:
			return spike_tensor # Empty tensor, don't raise an error

		binDuration = (ev_ts_us[-1] - ev_ts_us[0]) / self.num_time_bins
		if binDuration == 0:
			return spike_tensor
		# print(f"binDuration: {binDuration}")
		time_idx = ((ev_ts_us - ev_ts_us[0]) / binDuration)
		# print(f"ev_ts_us[0]: {ev_ts_us[0]}")
		# print(f"ev_ts_us: {ev_ts_us}")
		# print(f"(ev_ts_us - ev_ts_us[0]): {(ev_ts_us - ev_ts_us[0])}")
		# print(f"time_idx: {time_idx}")
		# Handle time stamps that are not floored and would exceed the allowed index after to-index conversion.
		time_idx[time_idx >= self.num_time_bins] = self.num_time_bins - 1

		spike_tensor[ev_pol.long(), ev_xy[:, 1].long(), ev_xy[:, 0].long(), time_idx.long()] = 1

		return spike_tensor
