#!/usr/bin/env python3
import argparse
from pathlib import Path

import torchviz
import gym

import rl
from rl.models import Estimator, PPO_mod, A2C_mod
from rl.models.utils import ActorCriticPolicy_mod

# ==================================================================================================
def get_model():
		env_id = "CartPole-events-sim-v1"
		env = gym.make(env_id)
		Estimator(env.observation_space)
		policy_kwargs = dict(features_extractor_class=Estimator)
		# model = PPO_mod(
		model = A2C_mod(
			ActorCriticPolicy_mod,
			env,
			policy_kwargs=policy_kwargs,
			device="cpu",
			n_steps=2,
			tensorboard_log=None,
			# pl_coef=0.0,
			# ent_coef=0.0,
			# vf_coef=0.0,
			# bs_coef=0.0,
			save_loss=True,
		)

		return model


# ==================================================================================================
def main(args: argparse.Namespace) -> None:
	model = get_model()
	params = model.policy.named_parameters()
	model.learn(total_timesteps=1)
	y = model.loss

	if args.verbose:
		graph = torchviz.make_dot(y.mean(), params=dict(params), show_attrs=True, show_saved=True)
	else:
		graph = torchviz.make_dot(y.mean(), params=dict(params))
	path = Path(__file__).parent / args.name
	graph.render(path.with_suffix(".gv"), format="png", outfile=path.with_suffix(".png"))


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("-v", "--verbose", action="store_true", help="Print more information")
	parser.add_argument("-n", "--name", type=str, help="Name of output files", default="gradients")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
