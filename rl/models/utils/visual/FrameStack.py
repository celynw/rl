from collections import deque

import gym

from .LazyFrames import LazyFrames

# ==================================================================================================
class FrameStack(gym.Wrapper):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env, k):
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

	# ----------------------------------------------------------------------------------------------
	def reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	# ----------------------------------------------------------------------------------------------
	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	# ----------------------------------------------------------------------------------------------
	def _get_ob(self):
		assert len(self.frames) == self.k
		return LazyFrames(list(self.frames))
