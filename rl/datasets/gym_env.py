
import argparse

import gym
from torch.utils.data.dataset import Dataset
from rich import print, inspect

# ==================================================================================================
class GymEnv(Dataset):
	"""EVReflex dataset with new extractor."""

	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, gym_env: str, algorithm):
		"""
		AI gym pytorch dataset.

		Args:
			args (argparse.Namespace): All arguments
			gym_env (str): Which environment type to use
		"""
		self.args = args
		self.env = gym.make(gym_env)
		self.algorithm = algorithm
		print(self.algorithm)

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
		group.add_argument("--timesteps", type=int, help="Total number of timesteps", default=1000)

		return parser

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		return self.args.timesteps

	# ----------------------------------------------------------------------------------------------
	def __getitem__(self, index: int) -> dict[str, object]:
		"""
		Get the dataset element at a specified index.

		Args:
			index (int): Index of dataset to retrieve

		Returns:
			dict[str, torch.Tensor]: Modified data structure
		"""

		# out = {
		# 	"spikes": spikes,
		# 	"flow": flows,
		# 	"img": imgs,
		# 	"index": index,
		# }



	# def learn(
	# 	self,
	# 	total_timesteps: int,
	# 	callback: MaybeCallback = None,
	# 	log_interval: int = 4,
	# 	eval_env: Optional[GymEnv] = None,
	# 	eval_freq: int = -1,
	# 	n_eval_episodes: int = 5,
	# 	tb_log_name: str = "run",
	# 	eval_log_path: Optional[str] = None,
	# 	reset_num_timesteps: bool = True,
	# ) -> "OffPolicyAlgorithm":

		rollout = self.algorithm.collect_rollouts(
			self.env,
			# train_freq=self.algorithm.train_freq,
			train_freq=4, # TODO
			# action_noise=self.algorithm.action_noise,
			action_noise=None, # TODO
			# callback=callback,
			callback=None,
			# learning_starts=self.algorithm.learning_starts,
			learning_starts=50000, # TODO
			# replay_buffer=self.algorithm.replay_buffer,
			replay_buffer=None, # TODO
			# log_interval=self.algorithm.,
			log_interval=4, # TODO
		)

		if index > 0 and index > self.learning_starts:
			# If no `gradient_steps` is specified,
			# do as many gradients steps as steps performed during the rollout
			gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
			# Special case when the user passes `gradient_steps=0`
			if gradient_steps > 0:
				self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

		return None
