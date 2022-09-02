from typing import Iterable
import optuna

# ==================================================================================================
class MultiplePruners(optuna.pruners.BasePruner):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, pruners: Iterable[optuna.pruners.BasePruner], pruning_condition: str = "any") -> None:
		self._pruners = tuple(pruners)

		self._pruning_condition_check_fn = None
		if pruning_condition == "any":
			self._pruning_condition_check_fn = any
		elif pruning_condition == "all":
			self._pruning_condition_check_fn = all
		else:
			raise ValueError(f"Invalid pruning ({pruning_condition}) condition passed!")
		assert self._pruning_condition_check_fn is not None

	# ----------------------------------------------------------------------------------------------
	def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
		return self._pruning_condition_check_fn(pruner.prune(study, trial) for pruner in self._pruners)
