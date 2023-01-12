#!/usr/bin/env python3
import gymnasium.spaces.box
import stable_baselines3.common.vec_env.stacked_observations
import torch

# ==================================================================================================
def main():
	os = gymnasium.spaces.box.Box(0, 1, (3, 480, 640))
	# env = gym.make("CartPole-v1")
	# env.reset()
	# obs, _, _, _, info = env.step(0)
	# SO = stable_baselines3.common.vec_env.stacked_observations.StackedObservations(1, 4, env.observation_space.shape)
	SO = stable_baselines3.common.vec_env.stacked_observations.StackedObservations(1, 4, os.shape)
	obs = torch.rand(os.shape)
	obs_, info_ = SO.update(obs, [], [])

	print(obs.shape)
	print(obs_.shape)


# ==================================================================================================
if __name__ == "__main__":
	# main()
	import gymnasium as gym
	import gymnasium.wrappers.frame_stack
	env = gym.make("CarRacing-v2")

	def get_shape():
		print(env.observation_space)
		# Box(4, 96, 96, 3)
		env.reset()
		obs, _, _, _, info = env.step([0, 0, 0])
		print(obs.shape)
		# (4, 96, 96, 3)

	get_shape()
	print()

	env = gym.wrappers.frame_stack.FrameStack(env, 4)
	get_shape()
