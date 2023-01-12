#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import time
from typing import Optional

import torch
import gymnasium as gym
import cv2
import numpy as np
from tqdm import tqdm
from rich import print, inspect

import rl
from rl.models import EDeNN
from rl.models.utils import PPO, ActorCriticPolicy
import rl.environments

# ==================================================================================================
def main(args: argparse.Namespace):
	env = gym.make(
		"CartPoleEvents-v0",
		args=args,
		return_rgb=True,
	)
	edenn = EDeNN(observation_space=env.observation_space, features_dim=env.state_space.shape[-1], projection_head=args.projection_head)

	obs = env.reset()
	args.outDir.mkdir(parents=True, exist_ok=True)
	for i in tqdm(range(args.steps or 1000000)):
		if args.cheat:
			action = int(((i / 25) / 2) > 0.5)
		else:
			features_extractor_kwargs = dict(features_dim=features_dim)
			policy_kwargs = dict(features_extractor_class=EDeNN, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs)
			model = PPO(
				ActorCriticPolicy,
				"CartPoleEvents-v0",
				policy_kwargs=policy_kwargs,
				verbose=1,
				device="cpu" if args.cpu else "auto",
			)
			action, _states = model.predict(obs, deterministic=True)
		events, reward, terminated, truncated, info = env.step(action)
		img = events_to_img(events)
		cv2.imwrite(str(args.outDir / f"{env.spec.name}_events_{i:04d}.png"), img)
		cv2.imwrite(str(args.outDir / f"{env.spec.name}_rgb_{i:04d}.png"), info["rgb"])

		# env.render()
		if terminated:
			if args.steps <= 0:
				break
			obs = env.reset()


# ==================================================================================================
def events_to_img(obs: np.ndarray):
	accumulated = obs.sum(1) if len(obs.shape) == 4 else obs
	img = np.zeros((3, *accumulated.shape[1:]))
	img[1] = accumulated[1]
	img[2] = accumulated[0]
	cv_img = np.transpose(img, [1, 2, 0])
	cv_img *= 255

	return cv_img


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	# parser.add_argument("env", type=str, help="Which gym environment to use", choices=["CartPole-events-v1", "MountainCar-events-v0", "Pong-events-v0", "CartPole-rgb" ,"MountainCar-rgb-v0", "Pong-rgb-v0"])
	parser.add_argument("outDir", type=Path, help="Output directory (will be created)")
	parser.add_argument("--model", type=Path, help="Model to load")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--steps", type=int, help="Number of steps, defaults to a single episode", default=0)
	parser.add_argument("--cheat", action="store_true", help="Simulate actions, don't load model")

	parser = EDeNN.add_argparse_args(parser)
	parser = rl.environments.CartPoleEvents.add_argparse_args(parser)

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())

# ./toys/demo_videos.py CartPole-events-v1 demo --cpu --steps 100 --model /vol/research/reflexive3/rl/SAVED/1663173675\ cartPole_NatureCNN_eventImg/best_model.zip --eventimg
# - Use NatureCNN
# ./toys/demo_videos.py MountainCar-events-v0 demo --cpu --steps 100 --model /vol/research/reflexive3/rl/SAVED/1663119160\ mountainCar_test/best_model.zip --projection_head 256 --cheat
# - Use EDeNNPH

# ffmpeg -framerate 30 -pattern_type glob -i "CartPole*_events_*.png" -c:v libx264 -pix_fmt yuv420p CartPole_events.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "CartPole*_rgb_*.png" -c:v libx264 -pix_fmt yuv420p CartPole_rgb.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "MountainCar*_events_*.png" -c:v libx264 -pix_fmt yuv420p MountainCar_events.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "MountainCar*_rgb_*.png" -c:v libx264 -pix_fmt yuv420p MountainCar_rgb.mp4
