#!/usr/bin/env python3
import argparse
from pathlib import Path
import inspect

import rl.models
import rl.environments

# ==================================================================================================
def parse_args() -> argparse.Namespace:
	"""
	Main argument parser.
	Models and environments also have specific arguments.
	They may also override the defaults for baseline hyperparameters, but only when not supplied.

	Returns:
		argparse.Namespace: Fully-parsed arguments object.
	"""
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
	models = [name for name, obj in inspect.getmembers(rl.models) if inspect.isclass(obj)]
	environments = [name for name, obj in inspect.getmembers(rl.environments) if inspect.isclass(obj)]

	# Add trainer-specific arguments ---------------------------------------------------------------
	group = parser.add_argument_group("Trainer")
	group.add_argument("name", type=str, metavar="EXPERIMENT_NAME", help="Name of experiment")
	group.add_argument("model", choices=models, metavar=f"MODEL: {{{', '.join(models)}}}", help="Model to train")
	group.add_argument("environment", choices=environments, metavar=f"ENVIRONMENT: {{{', '.join(environments)}}}", help="Environment to train on")
	parser.add_argument("-d", "--log_dir", type=Path, default=Path("/tmp/gym/"), help="Location of log directory")
	group.add_argument("--optuna", type=str, help="Optimise with optuna using this storage URL. Examples: 'sqlite:///optuna.db' or 'postgresql://postgres:password@host:5432/postgres'")
	# parser.add_argument("-r", "--render", action="store_true", help="Render final trained model output")
	parser.add_argument("-s", "--steps", type=int, default=1000, help="How many steps to train for")
	parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps before each weights update")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
	# parser.add_argument("--load_mlp", action="store_true", help="Load weights for the actor/critic")
	# parser.add_argument("--load_feat", action="store_true", help="Load weights for the feature extractor")
	parser.add_argument("--cpu", action="store_true", help="Force running on CPU")
	parser.add_argument("--gae_lambda", type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator", default=0.95)
	parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
	parser.add_argument("--n_epochs", type=int, help="Number of epoch when optimizing the surrogate loss", default=10)
	parser.add_argument("--ent_coef", type=float, help="Entropy coefficient for the loss calculation", default=0.0)
	parser.add_argument("--batch_size", type=int, help="Minibatch size", default=64)

	# Add model-specific arguments -----------------------------------------------------------------
	args_known, _ = parser.parse_known_args()
	Model = getattr(rl.models, args_known.model)
	parser = Model.add_argparse_args(parser) # Will chain to all child parser groups

	# Add environment-specific arguments -----------------------------------------------------------
	Environment = getattr(rl.environments, args_known.environment)
	parser = Environment.add_argparse_args(parser) # Will chain to all child parser groups

	try: # If we didn't add the help already (because an essential option was missing), add it now
		parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
	except argparse.ArgumentError:
		pass

	return parser.parse_args()
