"""
Based on MountainCar-v0 from gym=0.25.1
"""
from typing import Optional

import numpy as np
import cv2
import torch
from gym import spaces
from gym.envs.classic_control.mountain_car import MountainCarEnv

from rl.envs.utils import EventEnv

# ==================================================================================================
class MountainCarEnvEvents(EventEnv, MountainCarEnv):
	state_shape = (2, ) # TODO unused for now
	# ----------------------------------------------------------------------------------------------
	def __init__(self, init_ros: bool = True, tsamples: int = 10, event_image: bool = False):
		self.screen_width_ = 160
		self.screen_height_ = 96
		EventEnv.__init__(self, self.screen_width_, self.screen_height_, init_ros, tsamples, event_image)
		MountainCarEnv.__init__(self, render_mode=None)

		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		if self.event_image:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
		else:
			self.observation_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))
			# self.observation_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb):
		# Crop
		# rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# Resize
		rgb = cv2.resize(rgb, (self.screen_width_, self.screen_height_), interpolation=cv2.INTER_AREA)

		return rgb

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		_, reward, terminated, truncated, _ = super().step(action)
		info = {
			"updatedPolicy": int(self.updatedPolicy),
			"state": self.state,
		}

		event_tensor = self.get_events()
		if self.event_image:
			event_tensor = event_tensor.sum(1)
			# event_tensor = event_tensor.bool().double() * 255

		if terminated:
			# We're not doing setting this to False immediately because the monitor only writes a line when an episode is done
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
