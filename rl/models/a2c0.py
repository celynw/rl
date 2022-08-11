import argparse
from multiprocessing.sharedctypes import Value
from typing import Optional
from collections import deque
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import optuna
from rich import print, inspect
import numpy as np

from rl.models import Base
from rl.models.utils import Actor
from rl.models.utils import ModelType
# from rl.datasets.utils import Memories
from rl.datasets.utils import RolloutBuffer, RolloutBufferSamples
from rl.datasets import RLOnPolicy
from rl.utils import Step, Dir

# ==================================================================================================
class Discrete(torch.nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, obs_size: int, n_actions: int, softmax: bool, hidden_size: int = 128):
		"""
		Simple MLP network.

		Args:
			obs_size (int): Observation/state size of the environment
			n_actions (int): Number of discrete actions available in the environment
			hidden_size (int, optional): Size of hidden layers. Defaults to 128.
		"""
		super().__init__()
		# TODO cleanup
		if softmax:
			self.net = torch.nn.Sequential(
				torch.nn.Linear(obs_size, 24),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(24, 12),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(12, n_actions),
				torch.nn.Softmax(),
			)
		else:
			self.net = torch.nn.Sequential(
				torch.nn.Linear(obs_size, 24),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(24, 12),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(12, n_actions),
			)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		return self.net(x.float())


# ==================================================================================================
class Visual(torch.nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, input_shape, num_of_actions):
		"""
		Simple MLP network.

		Args:
			obs_size (int): Observation/state size of the environment
			n_actions (int): Number of discrete actions available in the environment
			hidden_size (int, optional): Size of hidden layers. Defaults to 128.
		"""
		raise NotImplementedError("A2C version not implemented yet")
		super().__init__()

		self.conv_nn = torch.nn.Sequential(
			torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU()
		)
		# Calculation of output of CNN so we can tell rest of NN what to expect on input.
		# `input_shape` had to be 1 dimension lower because we dont know size of that dim upfornt
		# So we need to add it every time if we want single frame to run through CNN
		# np.prod flattens output by product of sizes of every dimension
		cnn_output_shape = self.conv_nn(torch.zeros(1, *input_shape))
		cnn_output_shape = int(np.prod(cnn_output_shape.size()))
		# Output of regular NN will be 1x6 where 6 stands for 6 actions and how much NN thinks each action is right one
		self.linear_nn = torch.nn.Sequential(
			torch.nn.Linear(cnn_output_shape, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, num_of_actions)
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		# Called with either one element to determine next action, or a batch
		# during optimization. Returns tensor([[left0exp,right0exp]...]).

		batch_size = x.size()[0] # Bacth size will be either 1 or BATCH_SIZE
		# We need to flatten result of CNN and `view` reshapes tensor to have `batch_size` rows and data/batch_size columns (that is -1)
		cnn_output = self.conv_nn(x).view(batch_size, -1)
		return self.linear_nn(cnn_output) # apply rest of NN


# ==================================================================================================
class A2C(Base):
	"""Basic A2C Model."""
	monitor = f"{Step.TRAIN}/mean_reward" # TODO
	monitor_dir = Dir.MAX
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None) -> None:
		super().__init__(args, trial)
		self.envs = [self.create_env() for _ in range(1)]

		if not hasattr(self.envs[0], "_max_episode_steps"):
			self.envs[0]._max_episode_steps = float("inf")

		n_actions = self.envs[0].action_space.n
		if self.model_type is ModelType.DISCRETE:
			obs_size = self.envs[0].observation_space.shape[0]
			self.actor_net = Discrete(obs_size, n_actions, softmax=True)
			self.critic_net = Discrete(obs_size, 1, softmax=False)
		elif self.model_type is ModelType.VISUAL:
			obs_size = self.envs[0].observation_space.shape
			self.actor_net = Discrete(obs_size, n_actions, softmax=True)
			self.critic_net = Discrete(obs_size, 1, softmax=False)
		else:
			raise NotImplementedError(self.model_type)
		# self.example_input_array = torch.zeros(obs_size)
		# self.example_input_array = torch.zeros([1, 210, 160, 3])

		# self.buffers = [Memories() for i in range(self.hparams.batch_size)]
		# self.buffers = Memories(self.hparams.batch_size)
		# DEBUG
		self.observation_space = self.envs[0].observation_space
		self.action_space = self.envs[0].action_space
		buffer_length: int = 5
		self.gamma: float = 0.99
		self.gae_lambda: float = 0.95
		# self.n_envs = self.envs[0].num_envs
		self.n_envs = 1 # DEBUG
		self.num_rollouts: int = 100
		self.rollout_buffer = RolloutBuffer(
			buffer_length,
			self.observation_space,
			self.action_space,
			gamma=self.gamma,
			gae_lambda=self.gae_lambda,
			n_envs=self.n_envs,
		)
		# self.actor = Actor(self.envs, self.buffers)

		# self.total_reward = 0.0
		# self.episode_reward = 0.0
		# self.rewards = deque(maxlen=self.hparams.epoch_length)

		# self.populate()
		self.reset()

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
		# group.add_argument("--replay_size", type=int, default=1000, help="Capacity of the replay buffer")
		# group.add_argument("--warm_start_steps", type=int, default=1000, help="Number of iterations for linear warmup") # TODO pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		group.add_argument("--sync_rate", type=int, default=10, help="How many frames used to update the target network")
		# TODO refactor EPS here
		# group.add_argument("--eps_start", type=float, default=1.0, help="Epsilon starting value")
		# group.add_argument("--eps_end", type=float, default=0.01, help="Epsilon final value")
		# group.add_argument("--eps_last_frame", type=int, default=1000, help="Which frame epsilon should stop decaying at")

		parser = RLOnPolicy.add_argparse_args(parser)

		return parser

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		"""
		Reset the enviornment
		"""
		self._last_obs = self.envs[0].reset()
		# self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)
		self._last_dones = np.zeros((1,), dtype=np.bool)

	# ----------------------------------------------------------------------------------------------
	def populate(self) -> None:
		# """Initialises the buffer with starting states so the dataloader has something to return."""
		"""Initialises the buffer with a starting state so the dataloader has something to return."""
		# for i in range(self.hparams.batch_size):
		# 	# self.actor.play_step(self.critic_net, epsilon=1.0)
		# 	self.actor.play_step(self.actor_net, self.critic_net, self.device)

		self.actor.play_step(self.actor_net, self.critic_net, self.device)

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

	# # ----------------------------------------------------------------------------------------------
	# def get_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
	# 	# states, actions, rewards, dones, next_states = batch
	# 	log_probs, values, rewards, dones = batch

	# 	# DEBUG
	# 	# print(f"states.shape: {states.shape}")
	# 	# policy = self.actor_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
	# 	# print(f"policy.shape: {policy.shape}")
	# 	# print(f"policy: {policy}")
	# 	policy = self.actor_net(states)
	# 	# action_prob = F.softmax(policy, dim=-1)
	# 	# value = self.critic_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
	# 	value = self.critic_net(states)
	# 	print(f"action_prob.shape: {action_prob.shape}")
	# 	print(f"action_prob: {action_prob}")
	# 	print(f"value.shape: {value.shape}")

	# 	dist = Categorical(policy)
	# 	action = dist.sample()
	# 	actor_loss = -dist.log_prob(action) * advantage.detach()

	# 	with torch.no_grad():
	# 		# DEBUG
	# 		print(f"next_states.shape: {next_states.shape}")
	# 		next_state_values = self.critic_net(next_states)
	# 		print(f"next_state_values.shape: {next_state_values.shape}")

	# 		next_state_values = self.critic_net(next_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
	# 		# next_state_values = self.target_net(next_states).max(1)[0]
	# 		next_state_values[dones] = 0.0
	# 		next_state_values = next_state_values.detach()

	# 	# expected_state_action_values = next_state_values * self.hparams.gamma + rewards
	# 	advantage = (1.0 - dones) * self.hparams.gamma * next_state_values - next_state_values + rewards
	# 	# state_action_values = value

	# 	actor_loss = -dist.log_prob(action) * advantage.detach()
	# 	critic_loss = advantage.pow(2).mean()

	# 	# return torch.nn.MSELoss()(state_action_values, expected_state_action_values)
	# 	return actor_loss, critic_loss

	# ----------------------------------------------------------------------------------------------
	def collect_rollouts(self) -> RolloutBufferSamples:
		"""Collect rollouts and put them into the RolloutBuffer."""
		assert self._last_obs is not None, "No previous observation was provided"
		with torch.no_grad():
			# Sample new weights for the state dependent exploration
			# if self.use_sde:
			# 	self.reset_noise(self.env.num_envs)

			self.buffer_length = 100 # DEBUG
			self.sde_sample_freq = -1 # DEBUG
			self.eval() # TODO are you sure?
			for i in range(self.buffer_length):
				# if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
				# 	# Sample a new noise matrix
				# 	self.reset_noise(self.env.num_envs)

				# Convert to pytorch tensor, let Lightning take care of any GPU transfer
				obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
				probs = self.actor_net(obs_tensor)
				dist = torch.distributions.Categorical(probs=probs)
				values = self.critic_net(obs_tensor).flatten()

				actions = dist.sample()
				log_probs = dist.log_prob(actions)

				# Rescale and perform action
				clipped_actions = actions.cpu().numpy()
				# Clip the actions to avoid out of bound error
				if isinstance(self.action_space, gym.spaces.Box):
					clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
				elif isinstance(self.action_space, gym.spaces.Discrete):
					clipped_actions = clipped_actions.astype(np.int32)

				new_obs, rewards, dones, infos = self.envs[0].step(clipped_actions)

				if isinstance(self.action_space, gym.spaces.Discrete):
					# Reshape in case of discrete action
					actions = actions.view(-1, 1)

				if not torch.is_tensor(self._last_dones):
					self._last_dones = torch.as_tensor(self._last_dones).to(device=obs_tensor.device)
				rewards = torch.as_tensor(rewards).to(device=obs_tensor.device)

				self.rollout_buffer.add(obs_tensor, actions, rewards, self._last_dones, values, log_probs)
				self._last_obs = new_obs
				self._last_dones = dones

			final_obs = torch.as_tensor(new_obs).to(device=self.device, dtype=torch.float32)
			dist, final_values = self(final_obs)
			samples = self.rollout_buffer.finalize(final_values, torch.as_tensor(dones).to(device=obs_tensor.device, dtype=torch.float32))

			self.rollout_buffer.reset()
		self.train()
		return samples
	# ----------------------------------------------------------------------------------------------
	def step(self, step, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> dict:
		# Step through environment with actor
		rewards, dones, new_states = self.actor.play_step(self.actor_net, self.critic_net, self.device)
		self.episode_reward += sum(rewards)

		# if done:
		if all(dones) or (self.global_step > 0 and self.global_step % self.hparams.sync_rate == 0):
			self.rewards.append(self.episode_reward)
			self.total_reward = self.episode_reward
			self.episode_reward = 0

			q_val = self.critic_net(torch.tensor(new_states, device=self.device)).detach().data.cpu().numpy() # last q_val
			print(f"q_val.shape: {q_val.shape}")
			gamma = 0.99 # TODO

			# Train
			# train(self.buffer, last_q_val)
			# values = torch.stack(self.buffers.values())
			values = self.buffers.values()
			print(f"values.shape: {values.shape}")
			# q_vals = np.zeros((len(self.buffers), 1))
			# q_vals = np.zeros((*values.shape[:-1], 1))
			# all_q_vals = np.zeros((*values.shape[:-1], 1))
			all_q_vals = np.zeros_like(values)
			print(f"all_q_vals.shape: {all_q_vals.shape}")

			# Target values are calculated backward
			# Important: handle correctly done states! For those, the target should be equal to the reward only
			# for i, (_, _, reward, done) in enumerate(self.buffers.reversed()):
			for i, (_, _, rewards, dones) in enumerate(self.buffers.reversed()):
				print(i, rewards, dones)
				# q_val = reward + gamma * q_val * (1.0 - done)
				q_vals = [reward + gamma * q_val * (1.0 - done) for reward, done in zip(rewards, dones)]
				print(f"q_vals: {q_vals}")
				# FIX this part here for parallel envs (batch_size > 1)
				all_q_vals[0, len(self.buffers.buffers[0]) - 1 - i] = q_vals[0] # Store values from the end to the beginning

			advantage = torch.Tensor(all_q_vals) - values
			print(f"advantage.shape: {advantage.shape}")

			critic_loss = advantage.pow(2).mean()
			# adam_critic.zero_grad()
			# critic_loss.backward()
			# adam_critic.step()
			actor_loss = (-torch.stack(memory.log_probs) * advantage.detach()).mean()
			# adam_actor.zero_grad()
			# actor_loss.backward()
			# adam_actor.step()
			loss = actor_loss + critic_loss

			# for buffer in self.buffers:
			# 	buffer.clear()
			self.buffers.clear()

			self.log(f"{step}/actor_loss", actor_loss)
			self.log(f"{step}/critic_loss", critic_loss)
			self.log(f"{step}/loss", loss)

		else:
			loss = None


		# # Soft update of target network
		# if self.global_step % self.hparams.sync_rate == 0:
		# 	self.target_net.load_state_dict(self.net.state_dict())

		self.log(f"{step}/total_reward", self.total_reward)
		if len(self.rewards):
			self.log(f"{step}/mean_reward", sum(self.rewards) / len(self.rewards))

		return loss

	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> dict:
		return self.step(Step.TRAIN, batch, batch_idx, optimizer_idx)

	# ----------------------------------------------------------------------------------------------
	# def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
	# 	if batch_idx == 0:
	# 		print("\nVALIDATION\n")
	# 	return self.step("val", batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def on_test_start(self) -> None:
		self.epoch_bar_id = self.trainer.progress_bar_callback._add_task(self.envs[0]._max_episode_steps, description="episode")
		return super().on_test_start()

	# ----------------------------------------------------------------------------------------------
	def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
		# Ignoring batch TODO don't even provide it to save time and GPU memory
		self.actor.reset()
		for i, _ in enumerate(iter(bool, True), start=1):
			if i == 0:
				self.trainer.progress_bar_callback.progress.reset(self.epoch_bar_id)
			# self.trainer.progress_bar_callback._update(self.epoch_bar_id, current=i, total=self.envs[0]._max_episode_steps)
			self.trainer.progress_bar_callback._update(self.epoch_bar_id, current=i, total=float("inf"))
			reward, done = self.actor.play_step(self.actor_net, 0, self.device)
			self.episode_reward += reward
			if done:
				self.rewards.append(self.episode_reward)
				self.total_reward = self.episode_reward
				self.episode_reward = 0

			if self.hparams.vis: # TODO render multiple in batch?
				self.envs[0].render()
				time.sleep(1 / 30)
			if done:
				break

		self.log(f"{Step.TEST}/total_reward", self.total_reward, on_step=True, on_epoch=False)
		self.log(f"{Step.TEST}/mean_reward", sum(self.rewards) / len(self.rewards), on_step=True, on_epoch=False)

	# ----------------------------------------------------------------------------------------------
	def on_test_end(self) -> None:
		if self.hparams.vis: # TODO close all if rendering all in batch
			self.envs[0].close()
		return super().on_test_end()

	# ----------------------------------------------------------------------------------------------
	def configure_optimizers(self):
		"""
		Set up optimisers.

		Returns:
			Tuple[list[torch.optim.Optimizer], list[object]]: Optimiser(s) and learning rate scheduler(s)
		"""
		actor_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, amsgrad=True)
		critic_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, amsgrad=True)

		return [actor_optimizer, critic_optimizer]

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		# return DataLoader(
		# 	# RLOnPolicy(self.buffers, self.hparams.batch_size),
		# 	RLOnPolicy(self.buffers),
		# 	batch_size=self.hparams.batch_size,
		# 	num_workers=self.hparams.workers,
		# )
		return RLOnPolicy(self)
	# ----------------------------------------------------------------------------------------------
	# def val_dataloader(self) -> DataLoader:
	# 	"""Initialize the Replay Buffer dataset used for retrieving experiences"""
	# 	return DataLoader(
	# 		# RLOnPolicy(self.buffers, self.hparams.batch_size),
	# 		RLOnPolicy(self.buffers),
	# 		batch_size=self.hparams.batch_size,
	#		num_workers=self.hparams.workers,
	# 	)
	# 	return RLOnPolicy(self)
	# ----------------------------------------------------------------------------------------------
	def test_dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences"""
		# return DataLoader(
		# 	# RLOnPolicy(self.buffers, self.hparams.batch_size),
		# 	RLOnPolicy(self.buffers),
		# 	batch_size=1,
		# 	num_workers=self.hparams.workers,
		# )
		return RLOnPolicy(self)
