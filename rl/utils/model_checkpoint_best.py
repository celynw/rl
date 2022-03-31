from pathlib import Path
import os
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint

# ==================================================================================================
class ModelCheckpointBest(ModelCheckpoint):
	"""Additionally automatically saves a symlink to the best model."""
	# ----------------------------------------------------------------------------------------------
	def on_save_checkpoint(self, *args, **kwargs) -> dict[str, Any]:
		try:
			os.symlink(Path(self.best_model_path).name, Path(self.best_model_path).parent / "best_")
			os.replace(Path(self.best_model_path).parent / "best_", Path(self.best_model_path).parent / "best")
		except FileNotFoundError:
			pass
		return super().on_save_checkpoint(*args, **kwargs)
