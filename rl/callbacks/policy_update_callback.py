from stable_baselines3.common.callbacks import BaseCallback
import gym

# ==================================================================================================
class PolicyUpdateCallback(BaseCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env: gym.Env):
		"""
		Simple callback to set a flag to `True` in environments so we can see when the policy was updated.

		Args:
			env (gym.Env): Gym environment.
		"""
		super().__init__()
		self.env = env

	# ----------------------------------------------------------------------------------------------
	def _on_rollout_end(self) -> None:
		"""Update flag at the end of every rollout."""
		self.env.set_updatedPolicy()

	# ----------------------------------------------------------------------------------------------
	def _on_step(self) -> bool:
		"""
		If the callback returns False, training is aborted early.

		Returns:
			bool: True.
		"""
		return True
