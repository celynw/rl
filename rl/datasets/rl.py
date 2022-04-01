import argparse

from torch.utils.data import IterableDataset

from rl.datasets.utils import ReplayBuffer

# ==================================================================================================
class RL(IterableDataset):
	"""
	Iterable Dataset containing the ReplayBuffer
	which will be updated with new experiences during training

	Args:
		buffer: replay buffer
		sample_size: number of experiences to sample at a time
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
		self.buffer = buffer
		self.sample_size = sample_size

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
	def __iter__(self) -> tuple:
		states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(dones)):
			yield states[i], actions[i], rewards[i], dones[i], new_states[i]

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		return len(self.sample_size)
