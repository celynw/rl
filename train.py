#!/usr/bin/env python3
import colored_traceback.auto

import torch
import optuna

from rl import Objective
from rl.utils import MultiplePruners, DelayedThresholdPruner, parse_args, print_study_stats

# ==================================================================================================
if __name__ == "__main__":
	args = parse_args()
	if args.optuna is not None:
		torch.set_num_threads(1)
		objective = Objective(args)()

		# For RandomSampler, MedianPruner is the best
		# For TPESampler (default), Hyperband is the best
		pruner = MultiplePruners((optuna.pruners.HyperbandPruner(), DelayedThresholdPruner(lower=10, max_prunes=3)))
		study = optuna.create_study(
			study_name=f"{args.name}",
			direction=optuna.study.StudyDirection.MAXIMIZE,
			storage=args.optuna,
			load_if_exists=True,
			pruner=pruner,
			sampler=optuna.samplers.GridSampler({"tsamples": list(range(1, 26, 1))})
		)
		study.optimize(Objective(args), n_trials=100, n_jobs=1, gc_after_trial=False)
		print_study_stats(study)
	else:
		Objective(args)()
