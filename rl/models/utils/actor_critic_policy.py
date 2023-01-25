from typing import Optional

import torch
from stable_baselines3.common.policies import ActorCriticPolicy as SB3_ACP
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rich import print, inspect

import rl.models

# ==================================================================================================
class ActorCriticPolicy(SB3_ACP):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, detach: bool, **kwargs):
		"""Subclass `ActorCriticPolicy` to handle gradient breaks for the feature extractor."""
		super().__init__(*args, **kwargs)
		self.detach = detach

		# Final time slices from output, len() == N layers
		self.prev_features = [torch.tensor([0]), torch.tensor([0]), torch.tensor([0])] # First dim length is `n_envs` (vectorised environments)
		self.prev_features_train = [torch.tensor([0]), torch.tensor([0]), torch.tensor([0])] # First dim length is `n_steps` (rollout buffer length)

		# DEBUG
		# self.debug_step1 = 0
		# self.debug_step2 = 0

		# self.debug_obs1 = []
		# # self.debug_conv_weight1 = []
		# # self.debug_decay_weight1 = []
		# # self.debug_bias1 = []
		# self.debug_features1 = []
		# self.debug_latent_pi1 = []
		# self.debug_latent_vf1 = []
		# self.debug_values1 = []
		# self.debug_actions1 = []
		# self.debug_log_prob1 = []

		# self.debug_obs2 = []
		# # self.debug_conv_weight2 = []
		# # self.debug_decay_weight2 = []
		# # self.debug_bias2 = []
		# self.debug_features2 = []
		# self.debug_latent_pi2 = []
		# self.debug_latent_vf2 = []
		# self.debug_values2 = []
		# self.debug_actions2 = []
		# self.debug_log_prob2 = []

	# ----------------------------------------------------------------------------------------------
	def extract_features(self, obs: torch.Tensor, resets: Optional[torch.Tensor] = None, train: bool = False, features_extractor: Optional[BaseFeaturesExtractor] = None, idx: Optional[int] = None) -> torch.Tensor:
		"""
		Preprocess the observation if needed and extract features.

		:param obs:
		:return:
		"""
		features_extractor = features_extractor or self.features_extractor

		if not self.share_features_extractor:
			raise NotImplementedError()

		assert features_extractor is not None, "No features extractor was set"
		preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
		if isinstance(features_extractor, rl.models.EDeNN):
			if train:
			# 	# input: 8, 2, 768, 84, 84
			# 	# features: 1024, 2
			# 	# prev_features[0]: 8, 32, _, 20, 20
			# 	self.prev_features_train = [f.to(self.device) for f in self.prev_features_train]
			# 	# Need to use 8x previous features, 128 elements apart, in the first dimension
			# 	# Also need to do the same each time, for each epoch in the policy update

			# 	features, self.prev_features_train = self.features_extractor(preprocessed_obs, self.prev_features_train)
			# 	# self.prev_features_train = [f.detach().clone() for f in self.prev_features]
			# 	self.prev_features_train = [f.detach().clone() for f in self.prev_features_train]
			# 	# features, _ = self.features_extractor(preprocessed_obs[:, :, :6], [torch.tensor([0], device=self.device), torch.tensor([0], device=self.device), torch.tensor([0], device=self.device)])

			# 	# Reset should apply to NEXT time, so we do it after this features_extractor run
			# 	# if resets is not None:
			# 	for i, r in enumerate(resets):
			# 		if r != 0:
			# 			# print(r)
			# 			for p in range(len(self.prev_features_train)):
			# 				self.prev_features_train[p][i] *= 0


				# # Reset prev_features (for each layer) for that vectorized env in batch
				# # Setting to zero works, it's multiplied by decay and the added to first bin
				# for i, done in enumerate(resets):
				# 	if done:
				# 		for f, _ in enumerate(self.prev_features_train):
				# 			self.prev_features_train[f][i] *= 0

				# input: 8, 2, 6, 84, 84
				# features: 8, 2
				# prev_features_train[0]: 8, 32, _, 20, 20
				self.prev_features_train = [f.to(self.device) for f in self.prev_features_train]
				if idx is not None:
					raise RuntimeError("Didn't expect to call `extract_features` in this way during policy training update")
				features, self.prev_features_train = features_extractor(preprocessed_obs, self.prev_features_train)
				self.prev_features_train = [f.detach().clone() for f in self.prev_features_train]

				# Reset prev_features (for each layer) for that vectorized env in batch
				# Setting to zero works, it's multiplied by decay and the added to first bin
				for i, done in enumerate(resets):
					if done:
						for f, _ in enumerate(self.prev_features_train):
							self.prev_features_train[f][i] *= 0

			else:
				# input: 8, 2, 6, 84, 84
				# features: 8, 2
				# prev_features[0]: 8, 32, _, 20, 20
				self.prev_features = [f.to(self.device) for f in self.prev_features]
				if idx is not None:
					features, _ = features_extractor(preprocessed_obs, [f[idx:idx + 1] for f in self.prev_features]) # Keeps dim
					# We don't care about assigning back to `prev_features`... because this should be a terminal_observation and we're about to reset `prev_features`
				else:
					features, self.prev_features = features_extractor(preprocessed_obs, self.prev_features)
				self.prev_features = [f.detach().clone() for f in self.prev_features]
		else:
			features = features_extractor(preprocessed_obs)

		return features

	# ----------------------------------------------------------------------------------------------
	# def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, resets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, resets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Evaluate actions according to the current policy, given the observations.

		Args:
			obs (torch.Tensor): Observation.
			actions (torch.Tensor): Actions.

		Returns:
			tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: estimated values, log likelihood of taking those actions, entropy of the action distribution, features or projection
		"""
		features = self.extract_features(obs, resets, train=True)
		if self.detach:
			features_detached = features.clone().detach()
			if self.share_features_extractor:
				latent_pi, latent_vf = self.mlp_extractor(features_detached)
			else:
				pi_features, vf_features = features_detached
				latent_pi = self.mlp_extractor.forward_actor(pi_features)
				latent_vf = self.mlp_extractor.forward_critic(vf_features)
		else:
			if self.share_features_extractor:
				latent_pi, latent_vf = self.mlp_extractor(features)
			else:
				pi_features, vf_features = features
				latent_pi = self.mlp_extractor.forward_actor(pi_features)
				latent_vf = self.mlp_extractor.forward_critic(vf_features)

		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)
		entropy = distribution.entropy()

		# DEBUG
		# print(f"{self.debug_step1} {self.debug_step2}: {obs.detach().clone().unique()}")
		# self.debug_obs2.append(obs.detach().clone())
		# self.debug_features2.append(features.detach().clone())
		# print(f"debug_features2: {self.debug_features2[0].shape} ({len(self.debug_features2)})")
		# self.debug_latent_pi2.append(latent_pi.detach().clone())
		# self.debug_latent_vf2.append(latent_vf.detach().clone())
		# self.debug_values2.append(values.detach().clone())
		# self.debug_actions2.append(actions.detach().clone())
		# self.debug_log_prob2.append(log_prob.detach().clone())
		# self.debug_conv_weight2.append(self.features_extractor.layer1[0].conv_weight.data.detach().clone())
		# self.debug_decay_weight2.append(self.features_extractor.layer1[0].decay_weight.data.detach().clone())
		# self.debug_bias2.append(self.features_extractor.layer1[0].bias.data.detach().clone())
		# self.debug_step2 += 1


		# if not self.detach: # FIX - remove?? replace with check for EDeNN and projection head? Actually, replace detach with that check in the first place?
		if not self.features_extractor.use_bootstrap:
			return values, log_prob, entropy # As in super()

		# `self.features_extractor` always feeds through its layers up to `layer_last` to produce `features`.
		# If using a projection head, `features_dim` will likely be larger, with `projection_head` being the original size of `features_dim`.
		# In this case, `self.features_extractor.project()` needs to be called manually, same with the loss based on that output.
		# Gradients are detached from the input to the RL policy model, so the `features_extractor` is only trained using this.
		# So `projection` is only used for `bs_loss`.
		if isinstance(self.features_extractor, rl.models.EDeNN) and self.features_extractor.projection_head is not None:
			projection = self.features_extractor.project(features)
			return values, log_prob, entropy, projection
		else:
			return values, log_prob, entropy, features

	# ----------------------------------------------------------------------------------------------
	# Workaround for EvalCallback
	def predict_values(self, obs: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor:
		"""
		Get the estimated values according to the current policy given the observations.

		:param obs: Observation
		:return: the estimated values.
		"""
		# Added to use self, not super
		features = self.extract_features(obs, self.vf_features_extractor, idx=idx)
		latent_vf = self.mlp_extractor.forward_critic(features)
		return self.value_net(latent_vf)

	# ----------------------------------------------------------------------------------------------
	# Workaround for EvalCallback
	def get_distribution(self, obs: torch.Tensor) -> Distribution:
		"""
		Get the current policy distribution given the observations.

		:param obs:
		:return: the action distribution.
		"""
		# Added to use self, not super
		features = self.extract_features(obs, features_extractor=self.pi_features_extractor)
		# Added to ignore new_prev_x return
		if isinstance(features, tuple):
			features = features[0]
		latent_pi = self.mlp_extractor.forward_actor(features)
		return self._get_action_dist_from_latent(latent_pi)
