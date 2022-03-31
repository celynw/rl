from collections import deque

import numpy as np

from rl.datasets.utils import Experience

# ==================================================================================================
class ReplayBuffer:
	"""
	Replay Buffer for storing past experiences allowing the agent to learn from them.

	Args:
		capacity: size of the buffer
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, capacity: int) -> None:
		self.buffer = deque(maxlen=capacity)

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> None:
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
	def sample(self, batch_size: int) -> tuple:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx].__dict__.values() for idx in indices])

		return (
			np.array(states),
			np.array(actions),
			np.array(rewards, dtype=np.float32),
			np.array(dones, dtype=bool),
			np.array(next_states)
		)
