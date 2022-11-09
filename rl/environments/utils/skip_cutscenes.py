import gym
import ale_py
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

# ==================================================================================================
class SkipCutscenesPong(gym.Wrapper):
	reset_frames: int = 59
	score_frames: int = 63
	# ----------------------------------------------------------------------------------------------
	def reset(self, **kwargs) -> GymObs:
		obs = self.env.reset(**kwargs)
		for _ in range(self.reset_frames):
			action = [ale_py.Action.NOOP] * self.num_envs if self.num_envs > 1 else ale_py.Action.NOOP
			obs, _, _, _ = self.env.step(action)

		return obs

	# ----------------------------------------------------------------------------------------------
	def step(self, action: np.ndarray | int) -> GymStepReturn:
		obs, reward, terminated, truncated, info = self.env.step(action)
		if reward != 0.0:
			for _ in range(self.score_frames):
				# NOTE: Player's paddle could have momentum during this period
				# NOTE: Opponent's paddle will move during this period
				obs, reward, terminated, truncated, info = self.env.step(ale_py.Action.NOOP)

		return obs, reward, terminated, truncated, info
