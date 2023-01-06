from typing import Optional, NamedTuple

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer as SB3_ROB
from stable_baselines3.common.vec_env import VecNormalize

# ==================================================================================================
class RolloutBufferSamples(NamedTuple):
	observations: torch.Tensor
	actions: torch.Tensor
	old_values: torch.Tensor
	old_log_prob: torch.Tensor
	advantages: torch.Tensor
	returns: torch.Tensor
	states: torch.Tensor
	resets: torch.Tensor


# ==================================================================================================
class RolloutBuffer(SB3_ROB):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, state_shape, **kwargs):
		"""Subclassed to include physics states."""
		self.physics_shape = state_shape # `__init__` calls `reset()`
		super().__init__(*args, **kwargs)
		self.states = None
		self.resets = None

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		super().reset()
		self.states = np.zeros((self.buffer_size, self.n_envs) + self.physics_shape, dtype=np.float32)
		self.resets = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

	# ----------------------------------------------------------------------------------------------
	def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
		data = (
			self.observations[batch_inds],
			self.actions[batch_inds],
			self.values[batch_inds].flatten(),
			self.log_probs[batch_inds].flatten(),
			self.advantages[batch_inds].flatten(),
			self.returns[batch_inds].flatten(),
			self.states[batch_inds],
			self.resets[batch_inds],
		)
		return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

	# ----------------------------------------------------------------------------------------------
	def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor, state: Optional[torch.Tensor], reset: Optional[torch.Tensor]) -> None:
		if state is not None:
			self.states[self.pos] = state.clone().cpu().numpy()
		else:
			self.states[self.pos] = state
		self.resets[self.pos] = reset.clone().cpu().numpy()
		super().add(obs, action, reward, episode_start, value, log_prob)
