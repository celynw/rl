from typing import Optional, NamedTuple, Generator

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer as SB3_ROB
from stable_baselines3.common.vec_env import VecNormalize
from rich import print, inspect

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
		# DEBUG
		# print(f"batch_inds: {batch_inds}")
		# print(f"observations: {self.observations.shape}")
		# print(f"actions: {self.actions.shape}")
		# print(f"values: {self.values.shape}")
		# print(f"log_probs: {self.log_probs.shape}")
		# print(f"advantages: {self.advantages.shape}")
		# print(f"returns: {self.returns.shape}")
		# print(f"states: {self.states.shape}")
		data = (
			self.observations[batch_inds],
			self.actions[batch_inds],
			self.values[batch_inds].flatten(),
			self.log_probs[batch_inds].flatten(),
			self.advantages[batch_inds].flatten(),
			self.returns[batch_inds].flatten(),
			self.states[batch_inds].flatten(),
			self.resets[batch_inds].flatten(),
			# DEBUG
			# self.observations,
			# self.actions,
			# self.values.flatten(),
			# self.log_probs.flatten(),
			# self.advantages.flatten(),
			# self.returns.flatten(),
			# self.states,
			# self.resets,
		)
		# print(f"observations: {self.observations[:, :, batch_inds].shape}")
		# print(f"observations: {self.observations[batch_inds].shape}")
		# print(f"actions: {self.actions[batch_inds].shape}")
		# print(f"values: {self.values[batch_inds].flatten().shape}")
		# print(f"log_probs: {self.log_probs[batch_inds].flatten().shape}")
		# print(f"advantages: {self.advantages[batch_inds].flatten().shape}")
		# print(f"returns: {self.returns[batch_inds].flatten().shape}")
		# print(f"states: {self.states[batch_inds].shape}")

		# print(f"observations: {self.observations.shape}")
		# print(f"actions: {self.actions.shape}")
		# print(f"values: {self.values.flatten().shape}")
		# print(f"log_probs: {self.log_probs.flatten().shape}")
		# print(f"advantages: {self.advantages.flatten().shape}")
		# print(f"returns: {self.returns.flatten().shape}")
		# print(f"states: {self.states.shape}")

		return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

	# ----------------------------------------------------------------------------------------------
	def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor, state: Optional[torch.Tensor], reset: Optional[torch.Tensor]) -> None:
		if state is not None:
			self.states[self.pos] = state.clone().cpu().numpy()
		else:
			self.states[self.pos] = state
		self.resets[self.pos] = reset.clone().cpu().numpy()
		super().add(obs, action, reward, episode_start, value, log_prob)

	# ----------------------------------------------------------------------------------------------
	def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
		assert self.full, ""
		# Prepare the data
		if not self.generator_ready:

			_tensor_names = [
				"observations",
				"actions",
				"values",
				"log_probs",
				"advantages",
				"returns",
				"states",
				"resets",
			]
			for tensor in _tensor_names:
				shape = self.__dict__[tensor].shape
				self.__dict__[tensor] = self.__dict__[tensor].reshape(shape[0] * shape[1], *shape[2:])
			self.generator_ready = True

		# Return everything, don't create minibatches
		if batch_size is None:
			batch_size = self.buffer_size * self.n_envs

		start_idx = 0
		while start_idx < self.buffer_size * self.n_envs:
			yield self._get_samples(range(start_idx, start_idx + batch_size))
			start_idx += batch_size

