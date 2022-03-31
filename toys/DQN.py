#!/usr/bin/env python3
""" Unravelling stable_baselines3 to work out what the run-order is"""
import colored_traceback.auto

from typing import Optional

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from tqdm import tqdm

# ==================================================================================================
def main():
	env = gym.make("CartPole-v0")

	model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000)
	learn(model, total_timesteps=100000, log_interval=4) # model.learn(total_timesteps=10000, log_interval=4)
	model.save("dqn_cartpole")

	del model # remove to demonstrate saving and loading

	model = DQN.load("dqn_cartpole")

	obs = env.reset()
	while True:
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			obs = env.reset()


# ==================================================================================================
def learn(model, total_timesteps: int, log_interval: int = 4):
	# Default args
	callback: MaybeCallback = None
	eval_env: Optional[GymEnv] = None
	eval_freq: int = -1
	n_eval_episodes: int = 5
	tb_log_name: str = "run"
	eval_log_path: Optional[str] = None
	reset_num_timesteps: bool = True

	total_timesteps, callback = model._setup_learn(
		total_timesteps,
		eval_env,
		callback,
		eval_freq,
		n_eval_episodes,
		eval_log_path,
		reset_num_timesteps,
		tb_log_name,
	)

	callback.on_training_start(locals(), globals())

	bar = tqdm(total=total_timesteps)
	while model.num_timesteps < total_timesteps:
		bar.update(log_interval)
		rollout = model.collect_rollouts(
			model.env,
			train_freq=model.train_freq,
			action_noise=model.action_noise,
			callback=callback,
			learning_starts=model.learning_starts,
			replay_buffer=model.replay_buffer,
			log_interval=log_interval,
		)

		if rollout.continue_training is False:
			break

		print(model.num_timesteps)
		quit(0)
		if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
			# If no `gradient_steps` is specified,
			# do as many gradients steps as steps performed during the rollout
			gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
			# Special case when the user passes `gradient_steps=0`
			if gradient_steps > 0:
				model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

	callback.on_training_end()

	return model


# ==================================================================================================
if __name__ == "__main__":
	main()
