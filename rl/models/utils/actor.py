"""For reinforcement learning."""
import torch
import numpy as np
import gym

from rl.datasets.utils import Memory, Memories

from typing import Any

# ==================================================================================================
class Actor: # TODO consolidate with Agent? Remember to update __init__.py
	"""
	Base Actor class handling the interaction with the environment

	Args:
		env: training environment
		buffer: replay buffer storing experiences
	"""
	# ----------------------------------------------------------------------------------------------
	# def __init__(self, env: gym.Env, buffers: list[Memories]) -> None:
	def __init__(self, envs: list[gym.Env], buffers: Memories) -> None:
		self.envs = envs
		self.buffers = buffers
		self.reset()

	# ----------------------------------------------------------------------------------------------
	def reset(self) -> None:
		"""Resets the environment and updates the state."""
		self.states = [env.reset() for env in self.envs]

	# ----------------------------------------------------------------------------------------------
	# def get_action(self, net: torch.nn.Module, epsilon: float, device: str) -> int:
	# def get_action(self, state: torch.Tensor, net: torch.nn.Module, device: str) -> tuple[torch.Tensor, Any]:
	def get_actions(self, states: list[int], net: torch.nn.Module, device: str) -> tuple[list[int], list[float]]:
		# q_values = net(state)
		# _, action = torch.max(q_values, dim=1)
		# action = int(action.item())
		probs = net(torch.tensor(states, device=device))
		# TODO find out if I can just do this over the full batch
		actions = []
		log_probs = []
		for probs_ in probs:
			dist = torch.distributions.Categorical(probs=probs_)
			action = dist.sample()
			log_probs.append(dist.log_prob(action))
			actions.append(int(action.item()))
		# action = int(action.item())

		# return action, dist.log_prob(action)
		# return int(action.item()), dist.log_prob(action)
		return actions, log_probs

	# ----------------------------------------------------------------------------------------------
	@torch.no_grad()
	# def play_step(self, net: torch.nn.Module, epsilon: float = 0.0, device: str = "cpu") -> tuple[float, bool]:
	# def play_step(self, actor_net: torch.nn.Module, critic_net: torch.nn.Module, device: str = "cpu") -> tuple[float, bool]:
	def play_step(self, actor_net: torch.nn.Module, critic_net: torch.nn.Module, device: str = "cpu") -> tuple[float, bool, torch.Tensor]:
		"""
		Carries out a single interaction step between the agent and the environment

		Args:
			net: DQN network
			device: current device

		Returns:
			reward, done
		"""
		# FIX verify each env does nothing successfully if already done

		# state = torch.tensor(np.array([self.states]), device=device)
		# states = self.states.to(device)
		# action, log_prob = self.get_actions(state, actor_net, device)
		actions, log_probs = self.get_actions(self.states, actor_net, device)
		# Do step in the environments
		new_states = []
		rewards = []
		dones = []
		for i in range(len(self.envs)):
			new_state, reward, done, _ = self.envs[i].step(actions[i]) # Make sure detached!
			new_states.append(new_state)
			rewards.append(reward)
			dones.append(done)

		# # self.buffer.append(Memory(self.state, action, reward, done, new_state))
		# for i, buffer in enumerate(self.buffers):
		# 	# buffer.append(Memory(log_prob, critic_net(state), reward, done))
		# 	buffer.append(Memory(log_prob, critic_net(states[i]), reward, done))

		# self.buffers.append(Memory(log_probs, critic_net(states), rewards, dones))
		memories = []
		for i in range(len(log_probs)):
			memories.append(Memory(log_probs[i], critic_net(torch.tensor(self.states[i], device=device)), rewards[i], dones[i]))
		self.buffers.append(memories)

		self.states = new_states
		# if all(dones):
		# 	self.reset()
		for i, done in enumerate(dones):
			if done:
				self.envs[i].reset()

		# return reward, done
		return rewards, dones, new_states
