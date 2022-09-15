"""
Based on MountainCar-v0 from gym=0.25.1
"""
import math
from typing import Optional

import numpy as np
import torch
from gym import spaces
from gym.envs.atari.environment import AtariEnv
from gym.error import DependencyNotInstalled

from rl.envs.utils import EventEnv

# ==================================================================================================
class AtariEnvEvents(EventEnv, AtariEnv):
	state_shape = (2, ) # TODO unused for now
	# ----------------------------------------------------------------------------------------------
	def __init__(self, init_ros: bool = True, tsamples: int = 10, event_image: bool = False):
		self.screen_width_ = 160
		self.screen_height_ = 210 - 35 - 15
		EventEnv.__init__(self, self.screen_width_, self.screen_height_, init_ros, tsamples, event_image)
		AtariEnv.__init__(self, game="pong", render_mode="rgb_array")
		self.screen_width = 160
		self.screen_height = 210 - 35 - 15

		# NOTE: I should normalise my observation space (well, both), but not sure how to for event tensor
		if self.event_image:
			self._obs_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
			# self._obs_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.screen_height_, self.screen_width_))
		else:
			self._obs_space = spaces.Box(low=0, high=1, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))
			# self._obs_space = spaces.Box(low=0, high=255, dtype=np.double, shape=(2, self.tsamples, self.screen_height_, self.screen_width_))

	# ----------------------------------------------------------------------------------------------
	def resize(self, rgb):
		# Crop
		# rgb = rgb[int(self.screen_height * 0.4):int(self.screen_height * 0.8), :, :]
		# Resize
		# rgb = cv2.resize(rgb, (self.screen_width_, self.screen_height_), interpolation=cv2.INTER_AREA)
		# SIMON - Crop
		rgb = rgb[35:-15, :, :]
		# SIMON - Naively convert to greyscale (shit way to do it but don't want extra imports)
		rgb[:, :, 0] = rgb[:, :, 0] / 3 + rgb[:, :, 1] / 3 + rgb[:, :, 2] / 3
		# # SIMON - Make it 3 channel again
		# rgb[:, :, 1] = rgb[:, :, 0]
		# rgb[:, :, 2] = rgb[:, :, 0]
		# CELYN - Reduce to 1 channel
		rgb = rgb[:, :, 0][..., None]
		# # SIMON - rescale contrast
		# rgb = rgb - np.min(rgb)
		# rgb = np.clip(rgb * (255 / np.max(rgb)), 0, 255).astype("uint8")
		# SIMON - rescale contrast
		rgb = rgb - 57
		rgb = np.clip(rgb * (255 / 89), 0, 255).astype("uint8")

		return rgb

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		_, reward, terminated, truncated, _ = super().step(action)
		info = {
			"updatedPolicy": int(self.updatedPolicy),
			# "state": self.state,
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
		# info = {"state": self.state}
		info = {}

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
