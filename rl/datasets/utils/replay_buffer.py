from collections import deque

import numpy as np
import torch

from rl.datasets.utils import Experience, Memory

# ==================================================================================================
class ReplayBuffer:
	"""
	Replay Buffer for storing past experiences allowing the agent to learn from them.

	Args:
		capacity: size of the buffer
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, capacity: int) -> None:
		self.buffer: deque[Experience] = deque(maxlen=capacity)

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		return len(self.buffer)

	# ----------------------------------------------------------------------------------------------
	def append(self, experience: Experience) -> None:
		"""
		Add experience to the buffer.

		Args:
			experience (Experience): state, action, reward, done, new_state
		"""
		self.buffer.append(experience)

	# ----------------------------------------------------------------------------------------------
	def sample(self, sample_size: int) -> tuple:
		indices = np.random.choice(len(self.buffer), sample_size, replace=False)
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx].__dict__.values() for idx in indices])

		return (
			np.array(states),
			np.array(actions),
			np.array(rewards, dtype=np.float32),
			np.array(dones, dtype=bool),
			np.array(next_states)
		)


# ==================================================================================================
class Memories: # TODO rename to something better
	# ----------------------------------------------------------------------------------------------
	# def __init__(self) -> None:
	def __init__(self, batch_size: int) -> None:
		# self.buffer = []
		self.batch_size = batch_size
		self.buffers = [[] for _ in range(self.batch_size)]

		# self.log_probs = []
		# self.values = []
		# self.rewards = []
		# self.dones = []

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		return self.batch_size # TODO make sure I don't call this incorrectly, maye rename or something

	# ----------------------------------------------------------------------------------------------
	# def append(self, log_prob, value, reward, done):
	# def append(self, memory: Memory):
	# 	self.buffer.append(memory)
	def append(self, memories: list[Memory]):
		assert len(memories) == len(self.buffers)
		for memory, buffer in zip(memories, self.buffers):
			buffer.append(memory)

	# ----------------------------------------------------------------------------------------------
	def clear(self):
		# self.buffer = []
		self.buffers = [[] for _ in range(self.batch_size)]

	# # ----------------------------------------------------------------------------------------------
	# def _zip(self):
	# 	return zip(self.log_probs,
	# 			self.values,
	# 			self.rewards,
	# 			self.dones)

	# # ----------------------------------------------------------------------------------------------
	# def __iter__(self):
	# 	for data in self._zip():
	# 		return data

	# # ----------------------------------------------------------------------------------------------
	# def reversed(self):
	# 	for data in list(self._zip())[::-1]:
	# 		yield data

	# ----------------------------------------------------------------------------------------------
	def values(self):
		values = []
		for buffer in self.buffers:
			values.append([m.value for m in buffer])

		return torch.tensor(values)

	# ----------------------------------------------------------------------------------------------
	def reversed(self): # TODO
		# for data in list(zip(self.log_probs, self.values, self.rewards, self.dones))[::-1]:
		# 	yield data

		for i in range(len(self.buffers[0]) - 1, 0, -1):
			yield ( # TODO do I want to run and return for multiple steps, rather than just one?
				[buffer[i].log_prob for buffer in self.buffers],
				[buffer[i].value for buffer in self.buffers],
				[buffer[i].reward for buffer in self.buffers],
				[buffer[i].done for buffer in self.buffers],
			)


	# ----------------------------------------------------------------------------------------------
	# def sample(self, sample_size: int) -> tuple: # TODO rename or use __iter__ or something
	def sample(self) -> tuple: # TODO rename or use __iter__ or something
		# indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		# log_probs, values, rewards, dones = zip(*[self.buffer[idx].__dict__.values() for idx in indices])

		# return (
		# 	np.array(log_probs),
		# 	np.array(values),
		# 	np.array(rewards, dtype=np.float32),
		# 	np.array(dones, dtype=bool),
		# )
		# return (
		# 	self.buffer[-1].log_prob,
		# 	self.buffer[-1].value,
		# 	self.buffer[-1].reward,
		# 	self.buffer[-1].done,
		# )
		# return (
		# 	np.array([self.buffer[-1].log_prob]),
		# 	np.array([self.buffer[-1].value]),
		# 	np.array([self.buffer[-1].reward]),
		# 	np.array([self.buffer[-1].done]),
		# )

		# TODO
		# print(f"self.buffers.shape: {self.buffers.shape}")
		# quit(0)

		return ( # TODO do I want to run and return for multiple steps, rather than just one?
			[buffer[-1].log_prob for buffer in self.buffers],
			[buffer[-1].value for buffer in self.buffers],
			[buffer[-1].reward for buffer in self.buffers],
			[buffer[-1].done for buffer in self.buffers],
		)


from gym import spaces
from typing import NamedTuple

# ==================================================================================================
def get_action_dim(action_space: spaces.Space) -> int:
	"""
	Get the dimension of the action space.
	:param action_space: (spaces.Space)
	:return: (int)
	"""
	if isinstance(action_space, spaces.Box):
		return int(np.prod(action_space.shape))
	elif isinstance(action_space, spaces.Discrete):
		# Action is an int
		return 1
	elif isinstance(action_space, spaces.MultiDiscrete):
		# Number of discrete actions
		return int(len(action_space.nvec))
	elif isinstance(action_space, spaces.MultiBinary):
		# Number of binary actions
		return int(action_space.n)
	else:
		raise NotImplementedError()

# ==================================================================================================
def get_obs_shape(observation_space: spaces.Space) -> tuple[int, ...]:
	"""
	Get the shape of the observation (useful for the buffers).
	:param observation_space: (spaces.Space)
	:return: (Tuple[int, ...])
	"""
	if isinstance(observation_space, spaces.Box):
		return observation_space.shape
	elif isinstance(observation_space, spaces.Discrete):
		# Observation is an int
		return (1,)
	elif isinstance(observation_space, spaces.MultiDiscrete):
		# Number of discrete features
		return (int(len(observation_space.nvec)),)
	elif isinstance(observation_space, spaces.MultiBinary):
		# Number of binary features
		return (int(observation_space.n),)
	else:
		raise NotImplementedError()


# ==================================================================================================
class RolloutBufferSamples(NamedTuple):
	observations: torch.Tensor
	actions: torch.Tensor
	old_values: torch.Tensor
	old_log_probs: torch.Tensor
	advantages: torch.Tensor
	returns: torch.Tensor


# ==================================================================================================
class BaseBuffer(object):
	"""
	Base class that represent a buffer (rollout or replay)
	:param buffer_size: (int) Max number of element in the buffer
	:param observation_space: (spaces.Space) Observation space
	:param action_space: (spaces.Space) Action space
	:param device: (Union[torch.device, str]) PyTorch device
		to which the values will be converted
	:param n_envs: (int) Number of parallel environments
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(
		self,
		buffer_size: int,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		n_envs: int = 1,
	):
		super().__init__()
		self.buffer_size = buffer_size
		self.observation_space = observation_space
		self.action_space = action_space
		self.obs_shape = get_obs_shape(observation_space)
		self.action_dim = get_action_dim(action_space)
		self.pos = 0
		self.full = False
		self.n_envs = n_envs

	# ----------------------------------------------------------------------------------------------
	def size(self) -> int:
		"""
		:return: (int) The current size of the buffer
		"""
		if self.full:
			return self.buffer_size
		return self.pos

	# ----------------------------------------------------------------------------------------------
	def add(self, *args, **kwargs) -> None:
		"""
		Add elements to the buffer.
		"""
		raise NotImplementedError()

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		"""
		Reset the buffer.
		"""
		self.pos = 0
		self.full = False


# ==================================================================================================
class RolloutBuffer(BaseBuffer):
	"""
	Rollout buffer used in on-policy algorithms like A2C/PPO.
	:param buffer_size: (int) Max number of element in the buffer
	:param observation_space: (spaces.Space) Observation space
	:param action_space: (spaces.Space) Action space
	:param device: (torch.device)
	:param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
		Equivalent to classic advantage when set to 1.
	:param gamma: (float) Discount factor
	:param n_envs: (int) Number of parallel environments
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(
		self,
		buffer_size: int,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		gamma: float = 0.99,
		gae_lambda: float = 1,
		n_envs: int = 1,
	):
		super().__init__(buffer_size, observation_space, action_space, n_envs=n_envs)
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.reset()

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		self.initialised = False
		super().reset()

	# ----------------------------------------------------------------------------------------------
	def finalize(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> RolloutBufferSamples:
		"""
		Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
		Adapted from Stable-Baselines PPO2.
		Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
		to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
		where R is the discounted reward with value bootstrap,
		set ``gae_lambda=1.0`` during initialization.
		:param last_values: (torch.Tensor) estimated value of the current state
			following the current policy.
		:param last_dones: (torch.Tensor) End of episode signal.
		"""
		assert self.full, "Can only finalize RolloutBuffer when RolloutBuffer is full"

		assert last_values.device == self.values.device, 'All value function outputs must be on same device'

		last_gae_lam = 0
		advantages = torch.zeros_like(self.rewards)
		for step in reversed(range(self.buffer_size)):
			if step == self.buffer_size - 1:
				next_non_terminal = 1.0 - last_dones
				next_values = last_values
			else:
				next_non_terminal = 1.0 - self.dones[step + 1]
				next_values = self.values[step + 1]
			delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
			last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
			advantages[step] = last_gae_lam
		returns = advantages + self.values

		self.observations = self.observations.view((-1, *self.observations.shape[2:]))
		self.actions = self.actions.view((-1, *self.actions.shape[2:]))
		self.rewards = self.rewards.flatten()
		self.values = self.values.flatten()
		self.log_probs = self.log_probs.flatten()
		advantages = advantages.flatten()
		returns = returns.flatten()

		return RolloutBufferSamples(self.observations, self.actions, self.values, self.log_probs, advantages, returns)

	# ----------------------------------------------------------------------------------------------
	def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor) -> None:
		"""
		:param obs: (torch.tensor) Observation
		:param action: (torch.tensor) Action
		:param reward: (torch.tensor)
		:param done: (torch.tensor) End of episode signal.
		:param value: (torch.Tensor) estimated value of the current state
			following the current policy.
		:param log_prob: (torch.Tensor) log probability of the action
			following the current policy.
		"""

		# Initialise the first time we add something, so we know which device to put things on
		if not self.initialised:
			self.observations = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=obs.device)
			self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), device=action.device)
			self.rewards = torch.zeros((self.buffer_size, self.n_envs), device=reward.device)
			self.dones = torch.zeros((self.buffer_size, self.n_envs), device=done.device)
			self.values = torch.zeros((self.buffer_size, self.n_envs), device=value.device)
			self.log_probs = torch.zeros((self.buffer_size, self.n_envs), device=log_prob.device)
			self.initialised = True

		self.observations[self.pos] = obs
		self.actions[self.pos] = action
		self.rewards[self.pos] = reward
		self.dones[self.pos] = done
		self.values[self.pos] = value
		self.log_probs[self.pos] = log_prob
		self.pos += 1
		if self.pos == self.buffer_size:
			self.full = True
