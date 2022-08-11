from collections import deque

import gym
import numpy as np
import cv2

from rl.models.utils.visual.LazyFrames import LazyFrames

# ==================================================================================================
class CartPoleRGBTemp(gym.ObservationWrapper, gym.Wrapper):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env):
	# def __init__(self, env, k):
		super().__init__(env)
		# WarpFrame
		self.height = 84
		self.width = int(84 * (self.screen_width / self.screen_height))
		# self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

		# ImageToPyTorch
		old_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

		# FrameStack
		# gym.Wrapper.__init__(self, env)
		# self.k = k
		# self.frames = deque([], maxlen=k)
		# shp = env.observation_space.shape
		# self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

		# ScaledFloatFrame
		# self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)

	# ----------------------------------------------------------------------------------------------
	def observation(self, frame):
		# WarpFrame
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
		frame = frame[:, :, None]

		# ImageToPyTorch
		frame = np.moveaxis(frame, 2, 0)

		# ScaledFloatFrame
		# frame = np.array(frame).astype(np.float32) / 255.0

		return frame

	# # ----------------------------------------------------------------------------------------------
	# # FrameStack
	# def reset(self):
	# 	ob = self.env.reset()
	# 	for _ in range(self.k):
	# 		self.frames.append(ob)
	# 	return self._get_ob()

	# # ----------------------------------------------------------------------------------------------
	# def step(self, action):
	# 	ob, reward, done, info = self.env.step(action)
	# 	self.frames.append(ob)
	# 	return self._get_ob(), reward, done, info

	# # ----------------------------------------------------------------------------------------------
	# def _get_ob(self):
	# 	assert len(self.frames) == self.k
	# 	return LazyFrames(list(self.frames))
