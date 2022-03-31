#!/usr/bin/env python3
# https://www.pytorchlightning.ai/blog/en-lightning-reinforcement-learning-building-a-dqn-with-pytorch-lightning
# https://gist.github.com/djbyrne/94487be5f83b1232f02bdee79f2619b9#file-enlightened_dqn-py
import argparse
from dataclasses import dataclass
from collections import deque
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import gym
import numpy as np
from rich import print, inspect

# ==================================================================================================
def main(hparams) -> None:
	model = DQNLightning(hparams)
	checkpointCallback = ModelCheckpointBest(
		monitor=f"train/total_reward",
		# auto_insert_metric_name=False,
		# filename=f"epoch={{epoch:02d}} total_reward={{total_reward:.2f}}",
		save_top_k=3,
		save_last=True,
		mode="max",
		every_n_train_steps=20,
		# verbose=True
	)

	trainer = pl.Trainer(
		gpus=1,
		max_epochs=10000,
		# max_epochs=1000,
		# max_epochs=100,
		# val_check_interval=20, # WARNING: increments only within an epoch and resets for new ones! Use ModelCheckpoint's every_n_train_steps instead
		callbacks=[checkpointCallback],
	)
	# trainer.fit(model)
	# trainer.test(ckpt_path="best")

	# checkpointCallback._ModelCheckpoint__resolve_ckpt_dir(trainer) # Name mangling...
	# trainer.test(model=model, ckpt_path=Path(checkpointCallback.dirpath) / "best")
	trainer.test(model=model, ckpt_path="/home/celyn/Work/dev/pytorch/rl/lightning_logs/version_0/checkpoints/best")


# ==================================================================================================
class ModelCheckpointBest(ModelCheckpoint):
	# ----------------------------------------------------------------------------------------------
	def on_save_checkpoint(self, *args, **kwargs) -> dict[str, Any]:
		try:
			os.symlink(self.best_model_path, Path(self.best_model_path).parent / "best_")
			os.replace(Path(self.best_model_path).parent / "best_", Path(self.best_model_path).parent / "best")
		except FileNotFoundError:
			pass
		return super().on_save_checkpoint(*args, **kwargs)


# ==================================================================================================
class DQN(torch.nn.Module):
	"""
	Simple MLP network

	Args:
		obs_size: observation/state size of the environment
		n_actions: number of discrete actions available in the environment
		hidden_size: size of hidden layers
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
		super(DQN, self).__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(obs_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, n_actions)
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		return self.net(x.float())


# ==================================================================================================
@dataclass
class Experience:
	"""For storing experience steps gathered in training."""
	state: torch.Tensor
	action: int
	reward: float
	done: bool
	new_state: torch.Tensor


# ==================================================================================================
class ReplayBuffer:
	"""
	Replay Buffer for storing past experiences allowing the agent to learn from them

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
		Add experience to the buffer

		Args:
			experience: tuple (state, action, reward, done, new_state)
		"""
		self.buffer.append(experience)

	# ----------------------------------------------------------------------------------------------
	def sample(self, batch_size: int) -> tuple:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx].__dict__.values() for idx in indices])

		return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
				np.array(dones, dtype=bool), np.array(next_states))


# ==================================================================================================
class RLDataset(IterableDataset):
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
	def __iter__(self) -> tuple:
		states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(dones)):
			yield states[i], actions[i], rewards[i], dones[i], new_states[i]


# ==================================================================================================
class Agent:
	"""
	Base Agent class handeling the interaction with the environment

	Args:
		env: training environment
		buffer: replay buffer storing experiences
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, env: gym.Env, buffer: ReplayBuffer) -> None:
		self.env = env
		self.buffer = buffer
		self.reset()

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		"""Resents the environment and updates the state."""
		self.state = self.env.reset()

	# ----------------------------------------------------------------------------------------------
	def get_action(self, net: torch.nn.Module, epsilon: float, device: str) -> int:
		"""
		Using the given network, decide what action to carry out
		using an epsilon-greedy policy

		Args:
			net: DQN network
			epsilon: value to determine likelihood of taking a random action
			device: current device

		Returns:
			action
		"""
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state = torch.tensor(np.array([self.state]))

			if device not in ["cpu"]:
				state = state.cuda(device)

			q_values = net(state)
			_, action = torch.max(q_values, dim=1)
			action = int(action.item())

		return action

	# ----------------------------------------------------------------------------------------------
	@torch.no_grad()
	def play_step(self, net: torch.nn.Module, epsilon: float = 0.0, device: str = "cpu") -> tuple[float, bool]:
		"""
		Carries out a single interaction step between the agent and the environment

		Args:
			net: DQN network
			epsilon: value to determine likelihood of taking a random action
			device: current device

		Returns:
			reward, done
		"""
		action = self.get_action(net, epsilon, device)
		# Do step in the environment
		new_state, reward, done, _ = self.env.step(action)
		exp = Experience(self.state, action, reward, done, new_state)
		self.buffer.append(exp)

		self.state = new_state
		if done:
			self.reset()

		return reward, done


# ==================================================================================================
class DQNLightning(pl.LightningModule):
	"""Basic DQN Model."""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace) -> None:
		super().__init__()
		# self.hparams = args
		# self.hparams.update(args)
		self.save_hyperparameters(args)

		self.env = gym.make(self.hparams.env)
		obs_size = self.env.observation_space.shape[0]
		n_actions = self.env.action_space.n

		self.net = DQN(obs_size, n_actions)
		self.target_net = DQN(obs_size, n_actions)

		self.buffer = ReplayBuffer(self.hparams.replay_size)
		self.agent = Agent(self.env, self.buffer)

		self.total_reward = 0.0
		self.episode_reward = 0.0

		self.populate(self.hparams.warm_start_steps)

	# ----------------------------------------------------------------------------------------------
	def populate(self, steps: int = 1000) -> None:
		"""
		Carries out several random steps through the environment to initially fill
		up the replay buffer with experiences

		Args:
			steps: number of random steps to populate the buffer with
		"""
		for i in range(steps):
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
	def dqn_mse_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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
	def step(self, step, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		"""
		Carries out a single step through the environment to update the replay buffer.
		Then calculates loss based on the minibatch recieved

		Args:
			batch: current mini batch of replay data
			batch_idx: batch number

		Returns:
			Training loss and log metrics
		"""
		device = self.get_device(batch)
		epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame)

		# step through environment with agent
		reward, done = self.agent.play_step(self.net, epsilon, device)
		self.episode_reward += reward

		# calculates training loss
		loss = self.dqn_mse_loss(batch)
		loss = loss.unsqueeze(0)

		if done:
			self.total_reward = self.episode_reward
			self.episode_reward = 0

		# Soft update of target network
		if self.global_step % self.hparams.sync_rate == 0:
			self.target_net.load_state_dict(self.net.state_dict())

		self.log(f"{step}/total_reward", torch.tensor(self.total_reward).to(device))
		self.log(f"{step}/reward", torch.tensor(reward).to(device))

		return loss

	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		return self.step("train", batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	# def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
	# 	if batch_idx == 0:
	# 		print("\nVALIDATION\n")
	# 	return self.step("val", batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		# states, actions, rewards, dones, next_states = batch
		for _ in range(20):
			self.agent.reset()
			while True:
				self.env.render()
				_, _, done, _ = self.env.step(self.env.action_space.sample()) # take a random action
				if done:
					break
		self.env.close()
		quit(0)

		return self.step("train", batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def configure_optimizers(self) -> list[torch.optim.Optimizer]:
		"""Initialize Adam optimizer"""
		return [torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)]

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		return DataLoader(
			RLDataset(self.buffer, self.hparams.episode_length),
			batch_size=self.hparams.batch_size,
			# num_workers=8,
		)
	# ----------------------------------------------------------------------------------------------
	# def val_dataloader(self) -> DataLoader:
	# 	"""Initialize the Replay Buffer dataset used for retrieving experiences"""
	# 	return DataLoader(
	# 		RLDataset(self.buffer, self.hparams.episode_length),
	# 		batch_size=self.hparams.batch_size,
	# 		# num_workers=8,
	# 	)
	# ----------------------------------------------------------------------------------------------
	def test_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		return DataLoader(
			RLDataset(self.buffer, self.hparams.episode_length),
			batch_size=self.hparams.batch_size,
			# num_workers=8,
		)

	# ----------------------------------------------------------------------------------------------
	def get_device(self, batch) -> str:
		"""Retrieve device currently being used by minibatch."""
		return batch[0].device.index if self.on_gpu else "cpu"


# ==================================================================================================
if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
	parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
	parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--sync_rate", type=int, default=10,
						help="how many frames do we update the target network")
	parser.add_argument("--replay_size", type=int, default=1000,
						help="capacity of the replay buffer")
	parser.add_argument("--warm_start_size", type=int, default=1000,
						help="how many samples do we use to fill our buffer at the start of training")
	parser.add_argument("--eps_last_frame", type=int, default=1000,
						help="what frame should epsilon stop decaying")
	parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
	parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
	parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
	parser.add_argument("--max_episode_reward", type=int, default=200,
						help="max episode reward in the environment")
	parser.add_argument("--warm_start_steps", type=int, default=1000,
						help="max episode reward in the environment")
	parser.add_argument("-v", "--version", type=str, help="Try to continue training from this version", default=None)

	args = parser.parse_args()
	try:
		args.version = int(args.version)
	except:
		pass

	main(args)

	# model = DQNLightning.load_from_checkpoint(PATH)
