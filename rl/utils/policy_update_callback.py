from stable_baselines3.common.callbacks import BaseCallback

# ==================================================================================================
class PolicyUpdateCallback(BaseCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env):
		super().__init__()
		self.env = env

	# ----------------------------------------------------------------------------------------------
	def _on_rollout_end(self) -> None:
		self.env.set_updatedPolicy()

	# ----------------------------------------------------------------------------------------------
	def _on_step(self):
		return True
