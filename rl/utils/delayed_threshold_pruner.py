from typing import Optional
import optuna

# ==================================================================================================
class DelayedThresholdPruner(optuna.pruners.ThresholdPruner):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, n_warmup_steps: int = 0, interval_steps: int = 1, max_prunes: int = 3) -> None:
		self.max_prunes = max_prunes
		super().__init__(lower, upper, n_warmup_steps, interval_steps)
		self.prunes = set()
		self.step = None

	# ----------------------------------------------------------------------------------------------
	def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
		# This is also called whenever trial.should_prune() is called!
		# So we should handle the counting properly
		should_prune = super().prune(study, trial)

		if should_prune:
			self.prunes.add(trial.last_step)
			print(f"Prunes increased to {len(self.prunes)}")
			if len(self.prunes) >= self.max_prunes:
				return True
		else:
			self.prunes = set()
			print("Prunes reset")

		return False
