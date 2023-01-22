import pathlib
import io
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, explained_variance, get_system_info, check_for_correct_spaces
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from rich import print, inspect

from rl.models.utils import RolloutBuffer
import rl.models
from rl.environments.utils import get_base_envs

# ==================================================================================================
class PPO(SB3_PPO):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, policy, env, pl_coef: float = 1.0, bs_coef: float = 1.0, save_loss: bool = False, **kwargs):
		"""Subclassed to include physics states in the rollout buffer and bootstrap loss."""
		super().__init__(policy, env, **kwargs)
		self.pl_coef = pl_coef
		self.bs_coef = bs_coef
		self.save_loss = save_loss

		try:
			state_shape = get_base_envs(env)[0].state_space.shape
		except AttributeError: # CartPole-v1
			state_shape = get_base_envs(env)[0].observation_space.shape

		self.normalize_advantage = False if self.n_envs == 1 else self.normalize_advantage

		self.rollout_buffer = RolloutBuffer(
			self.n_steps,
			self.observation_space,
			self.action_space,
			device=self.device,
			gamma=self.gamma,
			gae_lambda=self.gae_lambda,
			n_envs=self.n_envs,
			state_shape=state_shape,
		)

	# # ----------------------------------------------------------------------------------------------
	# @classmethod
	# def load(
	# 	self,
	# 	path: str | pathlib.Path, io.BufferedIOBase,
	# 	state_shape,
	# 	env: Optional[GymEnv] = None,
	# 	device: torch.device | str = "auto",
	# 	custom_objects: Optional[Dict[str, Any]] = None,
	# 	print_system_info: bool = False,
	# 	force_reset: bool = True,
	# 	**kwargs,
	# ):
	# 	# super().load(cls, path, env, device, custom_objects, print_system_info, force_reset)
	# 	if print_system_info:
	# 		print("== CURRENT SYSTEM INFO ==")
	# 		get_system_info()

	# 	data, params, pytorch_variables = load_from_zip_file(
	# 		path, device=device, custom_objects=custom_objects, print_system_info=print_system_info
	# 	)

	# 	# Remove stored device information and replace with ours
	# 	if "policy_kwargs" in data:
	# 		if "device" in data["policy_kwargs"]:
	# 			del data["policy_kwargs"]["device"]

	# 	# print(data["policy_kwargs"])
	# 	# print(kwargs)
	# 	# print(kwargs["policy_kwargs"])

	# 	fek = data["policy_kwargs"]["features_extractor_kwargs"]
	# 	if "features_out" in fek:
	# 		print("### DEBUG! ###")
	# 		print("# Have to rename args to load from trained weights from an old code version!!")
	# 		fek["features_pre"] = fek["features_dim"]
	# 		fek["features_dim"] = fek["features_out"]
	# 		# fek["no_pre"] = False

	# 		del fek["features_out"]
	# 		data["policy_kwargs"]["features_extractor_kwargs"] = fek


	# 	# if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
	# 	# 	raise ValueError(
	# 	# 		f"The specified policy kwargs do not equal the stored policy kwargs."
	# 	# 		f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
	# 	# 	)

	# 	if "observation_space" not in data or "action_space" not in data:
	# 		raise KeyError("The observation_space and action_space were not given, can't verify new environments")

	# 	if env is not None:
	# 		# Wrap first if needed
	# 		env = cls._wrap_env(env, data["verbose"])
	# 		# Check if given env is valid
	# 		check_for_correct_spaces(env, data["observation_space"], data["action_space"])
	# 		# Discard `_last_obs`, this will force the env to reset before training
	# 		# See issue https://github.com/DLR-RM/stable-baselines3/issues/597
	# 		if force_reset and data is not None:
	# 			data["_last_obs"] = None
	# 	else:
	# 		# Use stored env, if one exists. If not, continue as is (can be used for predict)
	# 		if "env" in data:
	# 			env = data["env"]

	# 	# # noinspection PyArgumentList
	# 	# model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
	# 	# 	policy=data["policy_class"],
	# 	# 	state_shape=state_shape,
	# 	# 	env=env,
	# 	# 	device=device,
	# 	# 	_init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
	# 	# )

	# 	# load parameters
	# 	self.__dict__.update(data)
	# 	self.__dict__.update(kwargs)
	# 	self._setup_model()

	# 	# put state_dicts back in place
	# 	self.set_parameters(params, exact_match=True, device=device)

	# 	# put other pytorch variables back in place
	# 	if pytorch_variables is not None:
	# 		for name in pytorch_variables:
	# 			# Skip if PyTorch variable was not defined (to ensure backward compatibility).
	# 			# This happens when using SAC/TQC.
	# 			# SAC has an entropy coefficient which can be fixed or optimized.
	# 			# If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
	# 			# otherwise it is initialized to `None`.
	# 			if pytorch_variables[name] is None:
	# 				continue
	# 			# Set the data attribute directly to avoid issue when using optimizers
	# 			# See https://github.com/DLR-RM/stable-baselines3/issues/391
	# 			# recursive_setattr(self, name + ".data", pytorch_variables[name].data)
	# 			recursive_setattr(self, name + ".data", pytorch_variables[name].data)

	# 	# Sample gSDE exploration matrix, so it uses the right device
	# 	# see issue #44
	# 	if self.use_sde:
	# 		self.policy.reset_noise()  # pytype: disable=attribute-error

	# ----------------------------------------------------------------------------------------------
	def train(self) -> None:
		"""
		Update policy using the currently gathered rollout buffer.
		Adapted from super to include `bs_loss`.
		"""
		# Switch to train mode (this affects batch norm / dropout)
		self.policy.set_training_mode(True)
		# Update optimizer learning rate
		self._update_learning_rate(self.policy.optimizer)
		# Compute current clip range
		clip_range = self.clip_range(self._current_progress_remaining)
		# Optional: clip range for the value function
		if self.clip_range_vf is not None:
			clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

		entropy_losses = []
		if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
			bs_losses = []
		pg_losses, value_losses = [], []
		clip_fractions = []

		continue_training = True

		# DEBUG
		self.policy.debug_obs2 = []
		self.policy.debug_conv_weight2 = []
		self.policy.debug_decay_weight2 = []
		self.policy.debug_bias2 = []
		self.policy.debug_features2 = []
		self.policy.debug_latent_pi2 = []
		self.policy.debug_latent_vf2 = []
		self.policy.debug_values2 = []
		self.policy.debug_actions2 = []
		self.policy.debug_log_prob2 = []

		self.policy.prev_features_train_orig = [f.detach().clone() for f in self.policy.prev_features_train]

		# train for n_epochs epochs
		for epoch in range(self.n_epochs):
			print(f"EPOCH {epoch}")
			self.policy.debug_step2 = 0 # DEBUG

			approx_kl_divs = []
			# Do a complete pass on the rollout buffer
			for rollout_data in self.rollout_buffer.get(batch_size=self.n_envs): # Sampling has to be done at this size later
				actions = rollout_data.actions
				if isinstance(self.action_space, gym.spaces.Discrete):
					# Convert discrete action from float to long
					actions = rollout_data.actions.long().flatten()

				# Re-sample the noise matrix because the log_std has changed
				if self.use_sde:
					self.policy.reset_noise(self.n_epochs)

				# For bootstrap loss
				if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
					values, log_prob, entropy, features = self.policy.evaluate_actions(rollout_data.observations, actions, rollout_data.resets)
				else:
					values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, rollout_data.resets)
				values = values.flatten()
				# Normalize advantage
				advantages = rollout_data.advantages
				if self.normalize_advantage:
					advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

				# ratio between old and new policy, should be one at the first iteration
				ratio = torch.exp(log_prob - rollout_data.old_log_prob)

				# clipped surrogate loss
				policy_loss_1 = advantages * ratio
				policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
				policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

				# Logging
				pg_losses.append(policy_loss.item())
				clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
				clip_fractions.append(clip_fraction)

				if self.clip_range_vf is None:
					# No clipping
					values_pred = values
				else:
					# Clip the different between old and new value
					# NOTE: this depends on the reward scaling
					values_pred = rollout_data.old_values + torch.clamp(
						values - rollout_data.old_values, -clip_range_vf, clip_range_vf
					)
				# Value loss using the TD(gae_lambda) target
				value_loss = F.mse_loss(rollout_data.returns, values_pred)
				value_losses.append(value_loss.item())

				# Entropy loss favor exploration
				if entropy is None:
					# Approximate entropy when no analytical form
					entropy_loss = -torch.mean(-log_prob)
				else:
					entropy_loss = -torch.mean(entropy)

				entropy_losses.append(entropy_loss.item())

				if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
					bs_loss = F.l1_loss(features, rollout_data.states) # Assumed getting a single step out of the buffer at a time

				if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
					loss = (policy_loss * self.pl_coef) + (entropy_loss * self.ent_coef) + (value_loss * self.vf_coef) + (bs_loss * self.bs_coef)
				else:
					loss = (policy_loss * self.pl_coef) + (entropy_loss * self.ent_coef) + (value_loss * self.vf_coef)
				if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
					bs_losses.append(bs_loss.item())

				if self.save_loss:
					self.loss = loss.clone()

				# Calculate approximate form of reverse KL Divergence for early stopping
				# see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
				# and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
				# and Schulman blog: http://joschu.net/blog/kl-approx.html
				with torch.no_grad():
					log_ratio = log_prob - rollout_data.old_log_prob
					approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
					approx_kl_divs.append(approx_kl_div)

				if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
					continue_training = False
					if self.verbose >= 1:
						print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
					break

				# Optimization step
				self.policy.optimizer.zero_grad()
				assert not torch.isnan(loss) # Catch before too late, doesn't seem to trip up the training
				loss.backward()
				# Clip grad norm
				torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.policy.optimizer.step()

			if not continue_training:
				break

			# Reset previous features (train) to what it was before training each epoch!
			self.policy.prev_features_train = [f.detach().clone() for f in self.policy.prev_features_train_orig]

		# DEBUG
		for i in [
			"obs",
			"features",
			"latent_pi",
			"latent_vf",
			"values",
			"actions",
			"log_prob",
			"conv_weight",
			"decay_weight",
			"bias"
		]:
			results = []
			# NOTE: debug_X_2 might be longer (on account of) multiple epochs, but it'll be ignored in this function; it will only compare those from the first one anyway
			for li, (l1, l2) in enumerate(zip(self.policy.__dict__[f'debug_{i}1'], self.policy.__dict__[f'debug_{i}2'])):
				if i == "obs":
					# results.append(l1 == l2)
					results.append(torch.allclose(l1.float(), l2))
				else:
					results.append(torch.allclose(l1, l2))
			# print(f"{i}: {len([r for r in results if r is True])} / {len(results)} -> {len([r for r in results if r is True]) == len(results)}")

		# print(f"obs: {self.debug_obs1.shape} to {self.debug_obs2.shape}")
		# print(self.debug_obs1[:2, :, :, 0, 0])
		# print(self.debug_obs2[:2, :, :, 0, 0])
		# print(f"features: {self.debug_features1.shape} to {self.debug_features2.shape}")
		# print(f"latent_pi: {self.debug_latent_pi1.shape} to {self.debug_latent_pi2.shape}")
		# print(f"latent_vf: {self.debug_latent_vf1.shape} to {self.debug_latent_vf2.shape}")
		# print(f"values: {self.debug_values1.shape} to {self.debug_values2.shape}")
		# print(f"actions: {self.debug_actions1.shape} to {self.debug_actions2.shape}")
		# print(f"log_prob: {self.debug_log_prob1.shape} to {self.debug_log_prob2.shape}")

		# print(f"obs: {(self.debug_obs1 == self.debug_obs2).all()}")
		# # print(f"obs: {torch.allclose(self.debug_obs1, self.debug_obs2)}")
		# print(f"conv_weight: {(self.debug_conv_weight1 == self.debug_conv_weight2).all()}")
		# # print(f"conv_weight: {torch.allclose(self.debug_conv_weight1, self.debug_conv_weight2)}")
		# print(f"decay_weight: {(self.debug_decay_weight1 == self.debug_decay_weight2).all()}")
		# # print(f"decay_weight: {torch.allclose(self.debug_decay_weight1, self.debug_decay_weight2)}")
		# print(f"bias: {(self.debug_bias1 == self.debug_bias2).all()}")
		# # print(f"bias: {torch.allclose(self.debug_bias1, self.debug_bias2)}")

		# print(f"features0: {torch.allclose(self.debug_features1[0], self.debug_features2[0])}")
		# print(f"features1: {torch.allclose(self.debug_features1[1], self.debug_features2[1])}")
		# print(f"features2: {torch.allclose(self.debug_features1[2], self.debug_features2[2])}")
		# print(f"latent_pi: {torch.allclose(self.debug_latent_pi1, self.debug_latent_pi2)}")
		# print(f"latent_vf: {torch.allclose(self.debug_latent_vf1, self.debug_latent_vf2)}")
		# print(f"values: {torch.allclose(self.debug_values1, self.debug_values2)}")
		# print(f"actions: {(self.debug_actions1 == self.debug_actions2).all()}")
		# print(f"log_prob: {torch.allclose(self.debug_log_prob1, self.debug_log_prob2)}")


		# print(f"obs: {self.debug_obs1.shape} to {self.debug_obs2.shape}")
		# print(f"features: {self.debug_features1.shape} to {self.debug_features2.shape}")
		# print(f"latent_pi: {self.debug_latent_pi1.shape} to {self.debug_latent_pi2.shape}")
		# print(f"latent_vf: {self.debug_latent_vf1.shape} to {self.debug_latent_vf2.shape}")
		# print(f"values: {self.debug_values1.shape} to {self.debug_values2.shape}")
		# print(f"actions: {self.debug_actions1.shape} to {self.debug_actions2.shape}")
		# print(f"log_prob: {self.debug_log_prob1.shape} to {self.debug_log_prob2.shape}")

		# print(f"obs: {(self.debug_obs1 == self.debug_obs2[:, :, :6]).all()}")
		# print(f"conv_weight: {(self.debug_conv_weight1 == self.debug_conv_weight2).all()}")
		# print(f"decay_weight: {(self.debug_decay_weight1 == self.debug_decay_weight2).all()}")
		# print(f"bias: {(self.debug_bias1 == self.debug_bias2).all()}")

		# print(f"features: {torch.allclose(self.debug_features1, self.debug_features2[::128])}")
		# print(f"features: {torch.allclose(self.debug_features1[0], self.debug_features2[0])}")
		# # preprocessed_obs_ = preprocess_obs(self.debug_obs1, self.observation_space, normalize_images=self.normalize_images)
		# preprocessed_obs_ = preprocess_obs(self.debug_obs2[:, :, :6], self.observation_space, normalize_images=self.normalize_images)
		# # preprocessed_obs_ = preprocess_obs(self.debug_obs2, self.observation_space, normalize_images=self.normalize_images)
		# features_, _ = self.features_extractor(preprocessed_obs_, [torch.tensor([0], device=self.device), torch.tensor([0], device=self.device), torch.tensor([0], device=self.device)])
		# # features_ = features_[::128]
		# print(f"features_: {torch.allclose(self.debug_features1, features_)}")
		# print(f"latent_pi: {torch.allclose(self.debug_latent_pi1, self.debug_latent_pi2[::128])}")
		# print(f"latent_vf: {torch.allclose(self.debug_latent_vf1, self.debug_latent_vf2[::128])}")
		# print(f"values: {torch.allclose(self.debug_values1, self.debug_values2[::128])}")
		# print(f"actions: {(self.debug_actions1 == self.debug_actions2[::128]).all()}")
		# print(f"log_prob: {torch.allclose(self.debug_log_prob1, self.debug_log_prob2[::128])}")
		# NOTE: log_prob won't work unless all others correct
		# print(self.debug_features1)
		# print(self.debug_features2[::128])
		# print(self.debug_conv_weight1.shape) # torch.Size([32, 2, 1, 8, 8])
		# print(self.debug_conv_weight1[0])
		# print(self.debug_conv_weight2[0])
		# print(self.debug_decay_weight1.shape) # torch.Size([32, 1, 1])
		# print(self.debug_decay_weight1[0])
		# print(self.debug_decay_weight2[0])
		# quit(0)

		self._n_updates += self.n_epochs
		explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

		# Logs
		self.logger.record("train/entropy_loss", np.mean(entropy_losses))
		if isinstance(self.policy.features_extractor, rl.models.EDeNN) and self.policy.features_extractor.projection_head is not None:
			self.logger.record("train/bootstrap_loss", np.mean(bs_losses))
		self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
		self.logger.record("train/value_loss", np.mean(value_losses))
		self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
		self.logger.record("train/clip_fraction", np.mean(clip_fractions))
		self.logger.record("train/loss", loss.item())
		self.logger.record("train/explained_variance", explained_var)
		if hasattr(self.policy, "log_std"):
			self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/clip_range", clip_range)
		if self.clip_range_vf is not None:
			self.logger.record("train/clip_range_vf", clip_range_vf)

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

		Adapted from super to include `state` in the rollout buffer.

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

		self.policy.debug_obs1 = []
		self.policy.debug_conv_weight1 = []
		self.policy.debug_decay_weight1 = []
		self.policy.debug_bias1 = []
		self.policy.debug_features1 = []
		self.policy.debug_latent_pi1 = []
		self.policy.debug_latent_vf1 = []
		self.policy.debug_values1 = []
		self.policy.debug_actions1 = []
		self.policy.debug_log_prob1 = []

		self.policy.debug_step1 = 0
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

			state = infos[0]["state"]
			if state is not None:
				state = torch.tensor(infos[0]["state"], device=self.device)[None, ...]
			reset = torch.tensor(dones, device=self.device)[None, ...]

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

			# Handle timeout by bootstrapping with value function
			# see GitHub issue #633
			# Basically:
			# - We've run the observation through the policy (includes feature extractor)
			# - Worked out its value, and action to take
			# - Ran that action through the environment step
			# - If that is done, the next observation is actually the first in a reset environment
			# - But we can still access that done observation with `terminal_observation`
			# - Here, if the episode timed out, we need to use the value of that state
			# - If using an EDeNN, we still need those `prev_features` one last time BEFORE we reset them
			for idx, done in enumerate(dones):
				if (
					done
					and infos[idx].get("terminal_observation") is not None
					and infos[idx].get("TimeLimit.truncated", False)
				):
					terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
					with torch.no_grad():
						terminal_value = self.policy.predict_values(terminal_obs, idx=idx)[0]
					rewards[idx] += self.gamma * terminal_value

			if isinstance(self.policy.features_extractor, rl.models.EDeNN):
				# Reset prev_features (for each layer) for that vectorized env in batch
				# Setting to zero works, it's multiplied by decay and the added to first bin
				for envNum, done in enumerate(dones):
					if done:
						for layer, _ in enumerate(self.policy.prev_features):
							self.policy.prev_features[layer][envNum] *= 0
			rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, state, reset)
			self._last_obs = new_obs
			self._last_episode_starts = dones

		with torch.no_grad():
			# Compute value for the last timestep
			values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

		rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

		callback.on_rollout_end()

		return True
