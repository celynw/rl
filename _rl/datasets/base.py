"""Base class for all datasets."""
from pathlib import Path
import argparse

4from torch.utils.data.dataset import Dataset
from reflexive_slayer.utils import Step

# ==================================================================================================
class Base(Dataset):
	"""Base class for dataset."""
	height = None
	width = None

	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, split: Step):
		"""
		Dataset (base class).

		Args:
			args (argparse.Namespace): All arguments
			split (Step): TRAIN, VAL or TEST
		"""
		self.args = args
		assert self.args.batch_size >= 1

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
		group.add_argument("dataset_dir", type=Path, help="LMDB dataset directory")
		# group.add_argument("-b", "--batch_size", type=int, help="Batch size", default=4)
		# group.add_argument("-a", "--batch_accumulation", type=int, help="Perform batch accumulation", default=1)
		# group.add_argument("-w", "--workers", type=int, help="Dataset workers, can use 0", default=8)

		return parser

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		"""Get number of elements in the dataset."""
		return len(self.window_stamps)
