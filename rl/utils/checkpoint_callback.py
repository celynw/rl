#!/usr/bin/env python3
from pathlib import Path
import shutil
from typing import Any

from stable_baselines3.common.callbacks import CheckpointCallback as SB3_CheckpointCallback

# ==================================================================================================
class CheckpointCallback(SB3_CheckpointCallback):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, keep_n: int = 5, **kwargs):
		super().__init__(*args, **kwargs)
		self.keep_n = keep_n
		self.checkpoints: list[str] = []
		self.replay_buffers: list[str] = []
		self.vec_normalizes: list[str] = []

	# ----------------------------------------------------------------------------------------------
	def _on_step(self) -> bool:
		if self.n_calls % self.save_freq == 0:
			model_path = self._checkpoint_path(extension="zip")
			self.model.save(model_path)
			self.checkpoints.append(model_path)
			if self.verbose >= 2:
				print(f"Saving model checkpoint to {model_path}")
			mp = Path(model_path)
			shutil.copy(mp, (mp.parent / "latest").with_suffix(mp.suffix))

			if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
				# If model has a replay buffer, save it too
				replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
				self.model.save_replay_buffer(replay_buffer_path)
				self.replay_buffers.append(replay_buffer_path)
				if self.verbose > 1:
					print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

			if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
				# Save the VecNormalize statistics
				vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
				self.model.get_vec_normalize_env().save(vec_normalize_path)
				self.vec_normalizes.append(vec_normalize_path)
				if self.verbose >= 2:
					print(f"Saving model VecNormalize to {vec_normalize_path}")

			if self.keep_n > 0 and len(self.checkpoints) > self.keep_n:
				p = Path(self.checkpoints.pop(0))
				p.unlink()
				if self.verbose >= 2:
					print(f"Removed {p}")
				if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
					p = Path(self.replay_buffers.pop(0))
					p.unlink()
					if self.verbose >= 2:
						print(f"Removed {p}")
				if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
					p = Path(self.vec_normalizes.pop(0))
					p.unlink()
					if self.verbose >= 2:
						print(f"Removed {p}")

		return True
