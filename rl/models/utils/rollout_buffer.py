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
		# print(f"observations: {self.observations[batch_inds].shape}")
		# print(f"actions: {self.actions[batch_inds].shape}")
		# print(f"values: {self.values[batch_inds].shape} -> {self.values[batch_inds].flatten().shape}")
		# print(f"log_probs: {self.log_probs[batch_inds].shape} -> {self.log_probs[batch_inds].flatten().shape}")
		# print(f"advantages: {self.advantages[batch_inds].shape} -> {self.advantages[batch_inds].flatten().shape}")
		# print(f"returns: {self.returns[batch_inds].shape} -> {self.returns[batch_inds].flatten().shape}")
		# print(f"states: {self.states[batch_inds].shape} -> {self.states[batch_inds].flatten().shape}")
		# print(f"resets: {self.resets[batch_inds].shape} -> {self.resets[batch_inds].flatten().shape}")
		# quit(0)
		data = (
			self.observations[batch_inds],
			self.actions[batch_inds],
			self.values[batch_inds],
			self.log_probs[batch_inds],
			self.advantages[batch_inds],
			self.returns[batch_inds],
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

	# ----------------------------------------------------------------------------------------------
	# def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
	# 	assert self.full, ""
	# 	indices = np.random.permutation(self.buffer_size * self.n_envs)
	# 	# Prepare the data
	# 	if not self.generator_ready:

	# 		_tensor_names = [
	# 			"observations",
	# 			"actions",
	# 			"values",
	# 			"log_probs",
	# 			"advantages",
	# 			"returns",
	# 			"states",
	# 			"resets",
	# 		]
	# 		for tensor in _tensor_names:
	# 			self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
	# 		self.generator_ready = True

	# 	# Return everything, don't create minibatches
	# 	if batch_size is None:
	# 		batch_size = self.buffer_size * self.n_envs

	# 	start_idx = 0
	# 	while start_idx < self.buffer_size * self.n_envs:
	# 		yield self._get_samples(indices[start_idx : start_idx + batch_size])
	# 		start_idx += batch_size
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

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def swap_and_flatten_obs(arr: np.ndarray) -> np.ndarray:
		# """
		# Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
		# to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
		# to [n_steps * n_envs, ...] (which maintain the order)

		# :param arr:
		# :return:
		# """
		# shape = arr.shape
		# if len(shape) < 3:
		# 	shape = shape + (1,)
		# return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

		shape = arr.shape
		if len(shape) < 3:
			shape = shape + (1,)
		# return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

		# arr = arr.permute(0, 3, 1, 2)
		# arr = arr.swapaxes(0, 3, 1, 2)
		# print(arr.shape)
		arr = arr.transpose(0, 3, 1, 2, *list(range(len(shape)))[4:])
		arr = arr.reshape(shape[0] * shape[3], shape[1], shape[2], *shape[4:])
		# arr = arr.permute(1, 2, 0)
		arr = arr.transpose(1, 2, 0, *list(range(len(shape) - 1))[3:])
		# print(arr.shape)

		return arr

	# # ----------------------------------------------------------------------------------------------
	# @staticmethod
	# def unflatten_and_swap_feat(arr: torch.Tensor, dim: int) -> torch.Tensor:
	# 	shape = arr.shape
	# 	# if len(shape) < 3:
	# 	# 	shape = shape + (1,)

	# 	# arr = arr.transpose(2, 0, 1, *list(range(len(shape) - 1))[3:])
	# 	print(arr.shape)
	# 	arr = arr.permute(2, 0, 1, *list(range(len(shape)))[3:])
	# 	print(arr.shape)
	# 	arr = arr.reshape(shape[0], dim, shape[1], shape[2], *shape[3:])
	# 	# arr = arr.reshape(shape[2], 1, shape[0], shape[2], *shape[3:])
	# 	# arr = arr.reshape(shape[2], 1, shape[0], shape[1], *shape[3:])
	# 	print(arr.shape)
	# 	# arr = arr.transpose(0, 2, 3, 1, *list(range(len(shape)))[4:])
	# 	# arr = arr.permute(0, 2, 3, 1, -1)
	# 	arr = arr.permute(0, 2, 3, 1, 4, 5)
	# 	print(arr.shape)

	# 	return arr
