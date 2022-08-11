#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import time

import gym
# from stable_baselines3 import A2C
from stable_baselines3 import PPO
# from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from rich import print, inspect

# ==================================================================================================
def main(args: argparse.Namespace) -> None:
	log_dir = Path(f"/tmp/gym/{int(time.time())}")
	log_dir.mkdir(parents=True, exist_ok=True)

	all_rewards = []
	env_ = gym.make("CartPole-v1")
	for i in tqdm(range(args.times)):
		env = Monitor(env_, str(log_dir), allow_early_resets=True)
		# model = A2C(MlpPolicy, env, verbose=1)
		model = PPO(
			MlpPolicy,
			env,
			verbose=0,
			# Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
			# Equivalent to classic advantage when set to 1.
			# https://arxiv.org/abs/1506.02438
			# To obtain Monte-Carlo advantage estimate
			# 	(A(s) = R - V(S))
			# where R is the sum of discounted reward with value bootstrap
			# (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization
			# gae_lambda=1.0,
			clip_range=0.0,
			# max_grad_norm=1.0,
		)
		model.learn(
			total_timesteps=args.steps,
		)
		all_rewards.append(env.get_episode_rewards())

		if args.save and i == 0:
			# model.save(log_dir / "a2c_cartpole")
			model.save(log_dir / "ppo_cartpole")

		if args.render and i == args.times - 1:
			obs = env.reset()
			for i in range(1000):
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)
				env.render()
				if done:
					obs = env.reset()
	env.close()

	all_rewards_ = -np.ones([len(all_rewards), max([len(x) for x in all_rewards])])
	for i, sub in enumerate(all_rewards):
		all_rewards_[i][0:len(sub)] = sub
	all_rewards = np.ma.MaskedArray(all_rewards_, mask=all_rewards_ < 0)

	mean = all_rewards.mean(axis=0)
	std = all_rewards.std(axis=0)
	ax = plt.gca()
	ax.set_ylim([0, 500]) # TODO get env max reward instead of hard coding
	ax.fill_between(
		range(all_rewards.shape[1]), # X
		# np.clip(mean - std, 0, None), # Max
		mean - std, # Max
		mean + std, # Min
		alpha=.5,
		linewidth=0
	)
	ax.plot(mean, linewidth=2)
	plt.savefig(f"{args.name}.png")


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("times", type=int, help="Times to run")
	parser.add_argument("-r", "--render", action="store_true", help="Render final trained model output")
	parser.add_argument("-s", "--steps", type=int, default=1000, help="How many steps to train for")
	parser.add_argument("-S", "--save", action="store_true", help="Save first trained model")
	parser.add_argument("-n", "--name", type=str, help="Name of plot file")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
