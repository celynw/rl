#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import random

from setproctitle import setproctitle
import gym
from tqdm import tqdm
import numpy as np
from rich import print, inspect
import cv2

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback

import rl
from rl.envs.cartpole_events import CartPoleEnvEvents

# ==================================================================================================
class Object():
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		setproctitle(Path(__file__).name)
		self.parse_args()
		self.run()

	# ----------------------------------------------------------------------------------------------
	def parse_args(self):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument("examples", type=int, help="Number of examples to generate")
		parser.add_argument("-o", "--out_dir", type=Path, default=Path(__file__).parent / "dataset", help="Output directory")

		self.args = parser.parse_args()

	# ----------------------------------------------------------------------------------------------
	def run(self):
		fw = max(4, len(str(self.args.examples)))
		self.args.out_dir.mkdir(parents=True, exist_ok=True)
		# env = gym.make("CartPole-v1")
		env = gym.make("CartPole-events-v1")
		# env = gym.make("CartPole-events-debug")

		# FIX For some reason, I can't get ESIM to respond to a rendered image unless I do this
		model = PPO(MlpPolicy, env, n_steps=2, batch_size=2)
		model.learn(total_timesteps=2)

		# | Num | Observation           | Min                  | Max                |
		# |-----|-----------------------|----------------------|--------------------|
		# | 0   | Cart Position         | -4.8                 | 4.8                |
		# | 1   | Cart Velocity         | -Inf                 | Inf                |
		# | 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
		# | 3   | Pole Angular Velocity | -Inf                 | Inf                |
		# Got these extra values by checking vanilla CartPole by stepping until it fails, use those numbers
		for i in tqdm(range(self.args.examples)):
			x = random.uniform(-env.x_threshold, env.x_threshold) # 2.4
			x_dot = random.uniform(-1.7851658, 1.7851658)
			theta = random.uniform(-env.theta_threshold_radians, env.theta_threshold_radians) # 0.20943951023931953
			theta_dot = random.uniform(-2.7895198, 2.7895198)
			env.reset()
			env.set_state((x, x_dot, theta, theta_dot)) # Don't set `env.state` directly, it won't fail but won't work!
			# Run twice so that the event images have something to go on
			obs = env.step(random.randint(0, 1))
			obs = env.step(random.randint(0, 1))
			obs = obs[0] # Only interested in the event stream
			np.savez(self.args.out_dir / f"{i:0{fw}d}.npz", obs=obs, gt=np.array(env.state))
			# rgb = env.render("rgb_array")
			# cv2.imwrite(str(self.args.out_dir / f"{i:0{fw}d}.png"), rgb)

		env.close()


# ==================================================================================================
def main():
	Object()


# ==================================================================================================
if __name__ == "__main__":
	main()
