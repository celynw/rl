"""For reinforcement learning."""
import torch
import numpy as np
import gym

from rl.datasets.utils import Experience, ReplayBuffer

# ==================================================================================================
class Agent:
	"""
	Base Agent class handling the interaction with the environment

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
		Using the given network, decide what action to carry out using an epsilon-greedy policy

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
			state = torch.tensor(np.array([self.state]), device=device)

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
