from datetime import datetime

import optuna

# ==================================================================================================
def datestr() -> str:
	return datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


# ==================================================================================================
def print_study_stats(study: optuna.Study):
	print(f"Best params so far: {study.best_params}")
	print(f"Number of finished trials: {len(study.trials)}")
	print("Best trial:")
	trial = study.best_trial
	print(f"  Value: {trial.value}")
	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")
	print("  User attrs:")
	for key, value in trial.user_attrs.items():
		print(f"    {key}: {value}")
