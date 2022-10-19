from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import EvalCallback
import gym
import optuna

# ==================================================================================================
class TrialEvalCallback(EvalCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(
		self,
		eval_env: gym.Env,
		trial: optuna.Trial,
		# callback_on_new_best: Optional[BaseCallback] = None,
		n_eval_episodes: int = 5,
		eval_freq: int = 10000,
		best_model_save_path: Optional[Path] = None,
		deterministic: bool = True,
		verbose: int = 0,
	):
		super().__init__(
			eval_env=eval_env,
			best_model_save_path=str(best_model_save_path),
			n_eval_episodes=n_eval_episodes,
			eval_freq=eval_freq,
			deterministic=deterministic,
			verbose=verbose,
		)
		self.trial = trial
		self.eval_idx = 0
		self.is_pruned = False

	# ----------------------------------------------------------------------------------------------
	def _on_step(self) -> bool:
		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			super()._on_step()

			# Seems that env is reset before, but not after eval
			# If train_env == eval_env, we should reset afterwards
			if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
				self.eval_env.reset()

			self.eval_idx += 1
			self.trial.report(self.last_mean_reward, self.eval_idx)
			# Prune trial if need
			if self.trial.should_prune():
				self.is_pruned = True
				return False

		return True
