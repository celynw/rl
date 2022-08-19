#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torchviz
import gym

import rl
from rl.models import Estimator

# ==================================================================================================
def main(args: argparse.Namespace) -> None:
	model = torch.nn.Sequential()
	model.add_module("W0", torch.nn.Linear(8, 16))
	model.add_module("tanh", torch.nn.Tanh())
	model.add_module("W1", torch.nn.Linear(16, 1))

	x = torch.randn(1, 8)
	y = model(x)


	env_id = "CartPole-events-v1"
	env = gym.make(env_id)
	model = Estimator(env.observation_space)
	x = torch.zeros([1, 2, 64, 240])
	y = model(x)

	if args.verbose:
		graph = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
	else:
		graph = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))
	path = Path(__file__).parent / args.name
	graph.render(path.with_suffix(".gv"), format="png", outfile=path.with_suffix(".png"))


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Print more information")
	parser.add_argument("-n", "--name", type=str, help="Name of output files", default="gradients")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
