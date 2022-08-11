#!/usr/bin/env python3
import gym

# ==================================================================================================
if __name__ == "__main__":
	from rich import print, inspect

	env = gym.make("CartPole-v0")
	print(f"env.reward_range: {env.reward_range}")
	print(f"env._max_episode_steps: {env._max_episode_steps}")
	env = gym.make("Acrobot-v1")
	print(f"env.reward_range: {env.reward_range}")
	print(f"env._max_episode_steps: {env._max_episode_steps}")
	# inspect(env, all=True)
