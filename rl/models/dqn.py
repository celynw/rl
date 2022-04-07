import argparse
from typing import Optional
from collections import deque

import torch
from torch.utils.data import DataLoader
import gym
import optuna

from rl.models import Base
from rl.models.utils import Agent
from rl.datasets.utils import ReplayBuffer
from rl.datasets import RL
from rl.utils import Step, Dir

# ==================================================================================================
class DQN_net(torch.nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
		"""
		Simple MLP network.

		Args:
			obs_size (int): Observation/state size of the environment
			n_actions (int): Number of discrete actions available in the environment
			hidden_size (int, optional): Size of hidden layers. Defaults to 128.
		"""
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(obs_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, n_actions)
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		return self.net(x.float())


# ==================================================================================================
class DQN(Base):
	"""Basic DQN Model."""
	monitor = f"{Step.TRAIN}/mean_reward" # TODO
	monitor_dir = Dir.MAX
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None) -> None:
		super().__init__(args, trial)
		self.env = gym.make(self.hparams.env)
		obs_size = self.env.observation_space.shape[0]
		n_actions = self.env.action_space.n
		self.example_input_array = torch.zeros(obs_size)

		self.net = DQN_net(obs_size, n_actions)
		self.target_net = DQN_net(obs_size, n_actions)

		self.buffer = ReplayBuffer(self.hparams.replay_size)
		self.agent = Agent(self.env, self.buffer)

		self.total_reward = 0.0
		self.episode_reward = 0.0
		self.rewards = deque(maxlen=self.hparams.epoch_length)

		self.populate()

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Add model-specific arguments.

		Args:
			parser (argparse.ArgumentParser): Main parser

		Returns:
			argparse.ArgumentParser: Modified parser
		"""
		group = parser.add_argument_group("Model")
		envs = [env_spec.id for env_spec in gym.envs.registry.all()]
		group.add_argument("env", choices=envs, metavar=f"ENV", help="AI gym environment to use")
		args_known, _ = parser.parse_known_args()
		if args_known.env is None:
			parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

		group.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
		group.add_argument("--epoch_length", type=int, default=200, help="How many experiences to sample per pytorch epoch")
		group.add_argument("--replay_size", type=int, default=1000, help="Capacity of the replay buffer")
		# group.add_argument("--warm_start_steps", type=int, default=1000, help="Number of iterations for linear warmup") # TODO pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		group.add_argument("--sync_rate", type=int, default=10, help="How many frames used to update the target network")
		# TODO refactor EPS here
		group.add_argument("--eps_start", type=float, default=1.0, help="Epsilon starting value")
		group.add_argument("--eps_end", type=float, default=0.01, help="Epsilon final value")
		group.add_argument("--eps_last_frame", type=int, default=1000, help="Which frame epsilon should stop decaying at")

		parser = RL.add_argparse_args(parser)

		return parser

	# ----------------------------------------------------------------------------------------------
	def populate(self) -> None:
		"""
		Carries out several random steps through the environment to initially fill
		up the replay buffer with experiences

		Args:
			steps: number of random steps to populate the buffer with
		"""
		for i in range(self.hparams.replay_size):
			self.agent.play_step(self.net, epsilon=1.0)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Passes in a state x through the network and gets the q_values of each action as an output

		Args:
			x: environment state

		Returns:
			q values
		"""
		return self.net(x)

	# ----------------------------------------------------------------------------------------------
	def dqn_mse_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
		"""
		Calculates the mse loss using a mini batch from the replay buffer

		Args:
			batch: current mini batch of replay data

		Returns:
			loss
		"""
		states, actions, rewards, dones, next_states = batch

		state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

		with torch.no_grad():
			next_state_values = self.target_net(next_states).max(1)[0]
			next_state_values[dones] = 0.0
			next_state_values = next_state_values.detach()

		expected_state_action_values = next_state_values * self.hparams.gamma + rewards

		return torch.nn.MSELoss()(state_action_values, expected_state_action_values)

	# ----------------------------------------------------------------------------------------------
	def step(self, step, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		"""
		Carries out a single step through the environment to update the replay buffer.
		Then calculates loss based on the minibatch recieved.

		Args:
			batch: current mini batch of replay data
			batch_idx: batch number

		Returns:
			Training loss and log metrics
		"""
		epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame)

		# Step through environment with agent
		reward, done = self.agent.play_step(self.net, epsilon, self.device)
		self.episode_reward += reward

		loss = self.dqn_mse_loss(batch)
		loss = loss.unsqueeze(0)

		if done:
			self.rewards.append(self.episode_reward)
			self.total_reward = self.episode_reward
			self.episode_reward = 0

		# Soft update of target network
		if self.global_step % self.hparams.sync_rate == 0:
			self.target_net.load_state_dict(self.net.state_dict())

		self.log(f"{step}/loss", loss)
		self.log(f"{step}/total_reward", self.total_reward)
		if len(self.rewards):
			self.log(f"{step}/mean_reward", sum(self.rewards) / len(self.rewards))

		return loss

	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		return self.step(Step.TRAIN, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	# def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
	# 	if batch_idx == 0:
	# 		print("\nVALIDATION\n")
	# 	return self.step("val", batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def on_test_start(self) -> None:
		self.epoch_bar_id = self.trainer.progress_bar_callback._add_task(self.env._max_episode_steps, description="episode")
		return super().on_test_start()

	# ----------------------------------------------------------------------------------------------
	def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		# Ignoring batch TODO don't even provide it to save time and GPU memory
		self.agent.reset()
		for i in range(self.env._max_episode_steps):
			if i == 0:
				self.trainer.progress_bar_callback.progress.reset(self.epoch_bar_id)
			self.trainer.progress_bar_callback._update(self.epoch_bar_id, current=i, total=self.env._max_episode_steps)
			reward, done = self.agent.play_step(self.net, 0, self.device)
			self.episode_reward += reward
			if done:
				self.rewards.append(self.episode_reward)
				self.total_reward = self.episode_reward
				self.episode_reward = 0

			self.env.render()
			if done:
				break

		self.log(f"{Step.TEST}/total_reward", self.total_reward, on_step=True, on_epoch=False)
		self.log(f"{Step.TEST}/mean_reward", sum(self.rewards) / len(self.rewards), on_step=True, on_epoch=False)

	# ----------------------------------------------------------------------------------------------
	def on_test_end(self) -> None:
		self.env.close()
		return super().on_test_end()

	# ----------------------------------------------------------------------------------------------
	def configure_optimizers(self):
		"""
		Set up optimisers.

		Returns:
			Tuple[list[torch.optim.Optimizer], list[object]]: Optimiser(s) and learning rate scheduler(s)
		"""
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, amsgrad=True)

		return [optimizer]

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		return DataLoader(
			RL(self.buffer, self.hparams.epoch_length),
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.workers,
		)
	# ----------------------------------------------------------------------------------------------
	# def val_dataloader(self) -> DataLoader:
	# 	"""Initialize the Replay Buffer dataset used for retrieving experiences"""
	# 	return DataLoader(
	# 		RL(self.buffer, self.hparams.epoch_length),
	# 		batch_size=self.hparams.batch_size,
	#		num_workers=self.hparams.workers,
	# 	)
	# ----------------------------------------------------------------------------------------------
	def test_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		return DataLoader(
			RL(self.buffer, self.hparams.epoch_length),
			batch_size=1,
			num_workers=self.hparams.workers,
		)
