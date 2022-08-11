import gym
import numpy as np

# ==================================================================================================
class ScaledFloatFrame(gym.ObservationWrapper):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env):
		gym.ObservationWrapper.__init__(self, env)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

	# ----------------------------------------------------------------------------------------------
	def observation(self, observation):
		# careful! This undoes the memory optimization, use
		# with smaller replay buffers only.
		# print(type(observation))
		return np.array(observation).astype(np.float32) / 255.0
