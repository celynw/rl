import numpy as np
import torch
import torch.nn.functional as F
import gym
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, explained_variance

from rl.models.utils import RolloutBuffer_mod

# ==================================================================================================
class A2C_mod(A2C):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, pl_coef: float = 1.0, bs_coef: float = 1.0, **kwargs):
		super().__init__(*args, **kwargs)
		self.pl_coef = pl_coef
		self.bs_coef = bs_coef

		buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer_mod
		self.rollout_buffer = buffer_cls(
			self.n_steps,
			self.observation_space,
			self.action_space,
			device=self.device,
			gamma=self.gamma,
			gae_lambda=self.gae_lambda,
			n_envs=self.n_envs,
		)

	# ----------------------------------------------------------------------------------------------
	def train(self) -> None:
		"""Update policy using the currently gathered rollout buffer (one gradient step over whole data)."""
		# Switch to train mode (this affects batch norm / dropout)
		self.policy.set_training_mode(True)

		# Update optimizer learning rate
		self._update_learning_rate(self.policy.optimizer)

		# This will only loop once (get all data in one go)
		for rollout_data in self.rollout_buffer.get(batch_size=None):

			actions = rollout_data.actions
			if isinstance(self.action_space, gym.spaces.Discrete):
				# Convert discrete action from float to long
				actions = actions.long().flatten()

			values, log_prob, entropy, features = self.policy.evaluate_actions(rollout_data.observations, actions)
			values = values.flatten()

			# Normalize advantage (not present in the original implementation)
			advantages = rollout_data.advantages
			if self.normalize_advantage:
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

			# Policy gradient loss
			policy_loss = -(advantages * log_prob).mean()

			# Value loss using the TD(gae_lambda) target
			value_loss = F.mse_loss(rollout_data.returns, values)

			# Entropy loss favor exploration
			if entropy is None:
				# Approximate entropy when no analytical form
				entropy_loss = -torch.mean(-log_prob)
			else:
				entropy_loss = -torch.mean(entropy)

			bs_loss = F.l1_loss(features, rollout_data.states.squeeze(1))

			loss = (policy_loss * self.pl_coef) + (entropy_loss * self.ent_coef) + (value_loss * self.vf_coef) + (bs_loss * self.bs_coef)

			# Optimization step
			self.policy.optimizer.zero_grad()
			loss.backward()

			# Clip grad norm
			torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
			self.policy.optimizer.step()

		explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

		self._n_updates += 1
		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/explained_variance", explained_var)
		self.logger.record("train/entropy_loss", entropy_loss.item())
		self.logger.record("train/bootstrap_loss", bs_loss.item())
		self.logger.record("train/policy_loss", policy_loss.item())
		self.logger.record("train/value_loss", value_loss.item())
		if hasattr(self.policy, "log_std"):
			self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

	# ----------------------------------------------------------------------------------------------
	def collect_rollouts(
		self,
		env: VecEnv,
		callback: BaseCallback,
		rollout_buffer: RolloutBuffer,
		n_rollout_steps: int,
	) -> bool:
		"""
		Collect experiences using the current policy and fill a ``RolloutBuffer``.
		The term rollout here refers to the model-free notion and should not
		be used with the concept of rollout used in model-based RL or planning.

		:param env: The training environment
		:param callback: Callback that will be called at each step
			(and at the beginning and end of the rollout)
		:param rollout_buffer: Buffer to fill with rollouts
		:param n_steps: Number of experiences to collect per environment
		:return: True if function returned with at least `n_rollout_steps`
			collected, False if callback terminated rollout prematurely.
		"""
		assert self._last_obs is not None, "No previous observation was provided"
		# Switch to eval mode (this affects batch norm / dropout)
		self.policy.set_training_mode(False)

		n_steps = 0
		rollout_buffer.reset()
		# Sample new weights for the state dependent exploration
		if self.use_sde:
			self.policy.reset_noise(env.num_envs)

		callback.on_rollout_start()

		while n_steps < n_rollout_steps:
			if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
				# Sample a new noise matrix
				self.policy.reset_noise(env.num_envs)

			with torch.no_grad():
				# Convert to pytorch tensor or to TensorDict
				obs_tensor = obs_as_tensor(self._last_obs, self.device)
				actions, values, log_probs = self.policy(obs_tensor)
			actions = actions.cpu().numpy()

			# Rescale and perform action
			clipped_actions = actions
			# Clip the actions to avoid out of bound error
			if isinstance(self.action_space, gym.spaces.Box):
				clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

			new_obs, rewards, dones, infos = env.step(clipped_actions)

			assert(len(infos) == 1)
			info = infos[0]
			state = torch.tensor(info["physicsState"], device=self.device)[None, ...]

			self.num_timesteps += env.num_envs

			# Give access to local variables
			callback.update_locals(locals())
			if callback.on_step() is False:
				return False

			self._update_info_buffer(infos)
			n_steps += 1

			if isinstance(self.action_space, gym.spaces.Discrete):
				# Reshape in case of discrete action
				actions = actions.reshape(-1, 1)

			# Handle timeout by bootstraping with value function
			# see GitHub issue #633
			for idx, done in enumerate(dones):
				if (
					done
					and infos[idx].get("terminal_observation") is not None
					and infos[idx].get("TimeLimit.truncated", False)
				):
					terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
					with torch.no_grad():
						terminal_value = self.policy.predict_values(terminal_obs)[0]
					rewards[idx] += self.gamma * terminal_value

			rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, state)
			self._last_obs = new_obs
			self._last_episode_starts = dones

		with torch.no_grad():
			# Compute value for the last timestep
			values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

		rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

		callback.on_rollout_end()

		return True
