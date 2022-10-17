#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
import time
from typing import Optional

import torch
import gym
import cv2
import numpy as np
from tqdm import tqdm
from rich import print, inspect

import rl
from rl.models import PPO_mod, A2C_mod, Estimator, EstimatorPH, EDeNN, EDeNNPH, NatureCNN# RLPTCNN
from rl.models.utils import ActorCriticPolicy_mod

# ==================================================================================================
def main(args: argparse.Namespace):
	env = gym.make(args.env, tsamples=args.tsamples, event_image=args.eventimg, return_rgb=True)

	extractor = EDeNNPH if args.projection_head else EDeNN
	# extractor = None if args.projection_head else NatureCNN
	no_pre = False
	if args.env in ["CartPole-events-v1", "CartPole-rgb"]:
		features_dim = 4
	elif args.env in ["MountainCar-events-v0", "MountainCar-rgb-v0"]:
		features_dim = 2
	elif args.env in ["Pong-events-v0", "Pong-rgb-v0"]:
		assert not args.projection_head
		features_dim = 512
		no_pre = True
	else:
		raise RuntimeError
	features_extractor_kwargs = dict(features_dim=features_dim, features_pre=256)
	if not args.projection_head: # FIX Maybe only avoid doing this because of loading from old code versions
		features_extractor_kwargs["no_pre"] = no_pre
	# features_extractor_kwargs = dict(features_dim=256, features_pre=4, no_pre=no_pre)
	policy_kwargs = dict(features_extractor_class=extractor, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs)
	# policy_kwargs = dict(features_extractor_class=extractor, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs, net_arch=[256, 256])
	# policy_kwargs = dict(features_extractor_class=extractor, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs, net_arch=[dict(vf=[64], pi=[64])])
	# policy_kwargs = dict(features_extractor_class=extractor, optimizer_class=torch.optim.Adam, features_extractor_kwargs=features_extractor_kwargs, net_arch=[dict(pi=[64, 256], vf=[64, 256])])
	state_shape = (features_dim, ) if args.env not in ["Pong-events-v0", "Pong-rgb-v0"] else (128, )
	model = PPO_mod(
	# model = A2C_mod(
		ActorCriticPolicy_mod,
		env,
		policy_kwargs=policy_kwargs,
		verbose=1,
		device="cpu" if args.cpu else "auto",
		# pl_coef=0.0,
		# ent_coef=0.0,
		# vf_coef=0.0,
		# bs_coef=0.0,
		state_shape=state_shape,
		) # total_timesteps will be at least n_steps (2048)
	# model.load(args.model, state_shape=state_shape, policy_kwargs=policy_kwargs)



	obs = env.reset()
	args.outDir.mkdir(parents=True, exist_ok=True)
	for i in tqdm(range(args.steps or 1000000)):
		if args.cheat:
			action = int(((i / 25) / 2) > 0.5)
		else:
			action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		img = events_to_img(obs)
		cv2.imwrite(str(args.outDir / f"{args.env}_events_{i:04d}.png"), img)
		cv2.imwrite(str(args.outDir / f"{args.env}_rgb_{i:04d}.png"), info["rgb"])

		# env.render()
		if done:
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
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("env", type=str, help="Which gym environment to use", choices=["CartPole-events-v1", "MountainCar-events-v0", "Pong-events-v0", "CartPole-rgb" ,"MountainCar-rgb-v0", "Pong-rgb-v0"])
	parser.add_argument("outDir", type=Path, help="Output directory (will be created)")
	parser.add_argument("--model", type=Path, help="Model to load")
	parser.add_argument("--tsamples", type=int, help="How many time samples to use in event environments", default=6)
	parser.add_argument("--eventimg", action="store_true", help="Whether to accumulate events into images")
	parser.add_argument("--projection_head", action="store_true", help="Use projection head")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--steps", type=int, help="Number of steps, defaults to a single episode", default=0)
	parser.add_argument("--cheat", action="store_true", help="Simulate actions, don't load model")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())

# ./toys/demo_videos.py CartPole-events-v1 demo --cpu --steps 100 --model /vol/research/reflexive3/rl/SAVED/1663173675\ cartPole_NatureCNN_eventImg/best_model.zip --eventimg
# - Use NatureCNN
# ./toys/demo_videos.py MountainCar-events-v0 demo --cpu --steps 100 --model /vol/research/reflexive3/rl/SAVED/1663119160\ mountainCar_test/best_model.zip --projection_head --cheat
# - Use EDeNNPH

# ffmpeg -framerate 30 -pattern_type glob -i "CartPole*_events_*.png" -c:v libx264 -pix_fmt yuv420p CartPole_events.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "CartPole*_rgb_*.png" -c:v libx264 -pix_fmt yuv420p CartPole_rgb.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "MountainCar*_events_*.png" -c:v libx264 -pix_fmt yuv420p MountainCar_events.mp4
# ffmpeg -framerate 30 -pattern_type glob -i "MountainCar*_rgb_*.png" -c:v libx264 -pix_fmt yuv420p MountainCar_rgb.mp4
