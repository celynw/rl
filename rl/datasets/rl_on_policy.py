import argparse
# from collections import deque

import torch
from torch.utils.data import IterableDataset

# from rl.datasets.utils import Memories
from rl.datasets.utils import RolloutBufferSamples # DEBUG

# # ==================================================================================================
# class RLOnPolicy(IterableDataset):
# 	"""
# 	Iterable Dataset containing the Memories
# 	which will be updated with new experiences during training

# 	Args:
# 		buffer: replay buffer
# 		sample_size: number of experiences to sample at a time
# 	"""
# 	# ----------------------------------------------------------------------------------------------
# 	# def __init__(self, buffers: list[Memories], batch_size: int = 16) -> None:
# 	# def __init__(self, batch_size: int = 16) -> None:
# 	# def __init__(self, memories: Memories) -> None:
# 	def __init__(self, model) -> None:
# 		# self.memories = []
# 		# self.memories = deque(maxlen=batch_size)
# 		# self.memories = memories
# 		self.model = model

# 	# ----------------------------------------------------------------------------------------------
# 	@staticmethod
# 	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
# 		"""
# 		Add model-specific arguments to parser.

# 		Args:
# 			parser (argparse.ArgumentParser): Main parser

# 		Returns:
# 			argparse.ArgumentParser: Modified parser
# 		"""
# 		group = parser.add_argument_group("Dataset")
# 		group.add_argument("-b", "--batch_size", type=int, help="Batch size", default=16)
# 		group.add_argument("-a", "--batch_accumulation", type=int, help="Perform batch accumulation", default=1)
# 		group.add_argument("-w", "--workers", type=int, help="Dataset workers, can use 0", default=0)

# 		return parser

# 	# ----------------------------------------------------------------------------------------------
# 	def __iter__(self) -> tuple:
# 		# log_probs, values, rewards, dones = self.buffer.sample(self.sample_size)
# 		for _ in range(len(self.memories)): # For each in batch_size
# 			yield self.memories.sample()

# 	# # ----------------------------------------------------------------------------------------------
# 	# def __len__(self) -> int:
# 	# 	return self.sample_size



# ==================================================================================================
class RLOnPolicy:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, model):
		self.model = model

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Add model-specific arguments to parser.

		Args:
			parser (argparse.ArgumentParser): Main parser

		Returns:
			argparse.ArgumentParser: Modified parser
		"""
		group = parser.add_argument_group("Dataset")
		group.add_argument("-b", "--batch_size", type=int, help="Batch size", default=16)
		group.add_argument("-a", "--batch_accumulation", type=int, help="Perform batch accumulation", default=1)
		group.add_argument("-w", "--workers", type=int, help="Dataset workers, can use 0", default=0)

		return parser

	# ----------------------------------------------------------------------------------------------
	def __iter__(self):
		for i in range(self.model.num_rollouts):
			experiences = self.model.collect_rollouts()
			observations, actions, old_values, old_log_probs, advantages, returns = experiences
			for j in range(self.model.epochs_per_rollout):
				k = 0
				perm = torch.randperm(observations.shape[0], device=observations.device)
				while k < observations.shape[0]:
					batch_size = min(observations.shape[0] - k, self.model.batch_size)
					yield RolloutBufferSamples(
						observations[perm[k:k + batch_size]],
						actions[perm[k:k + batch_size]],
						old_values[perm[k:k + batch_size]],
						old_log_probs[perm[k:k + batch_size]],
						advantages[perm[k:k + batch_size]],
						returns[perm[k:k + batch_size]])
					k += batch_size
