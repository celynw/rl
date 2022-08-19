from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

# ==================================================================================================
class TqdmCallback(BaseCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		super().__init__()
		self.progress_bar = None

	# ----------------------------------------------------------------------------------------------
	def _on_training_start(self):
		self.progress_bar = tqdm(total=self.locals["total_timesteps"], position=1, leave=True)

	# ----------------------------------------------------------------------------------------------
	def _on_step(self):
		self.progress_bar.update(1)
		return True

	# ----------------------------------------------------------------------------------------------
	def _on_training_end(self):

		self.progress_bar.close()
		self.progress_bar = None
