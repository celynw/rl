from typing import Optional

import torch
from stable_baselines3.common.policies import ActorCriticPolicy as SB3_ACP
from stable_baselines3.common.preprocessing import preprocess_obs
from rich import print, inspect

import rl.models

# ==================================================================================================
class ActorCriticPolicy(SB3_ACP):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, *args, detach: bool = True, **kwargs):
		"""Subclass `ActorCriticPolicy` to handle gradient breaks for the feature extractor."""
		super().__init__(*args, **kwargs)
		self.detach = detach

		# Final time slices from output, len() == N layers
		self.prev_features = [torch.tensor([0]), torch.tensor([0]), torch.tensor([0])] # First dim length is `n_envs` (vectorised environments)
		self.prev_features_train = [torch.tensor([0]), torch.tensor([0]), torch.tensor([0])] # First dim length is `n_steps` (rollout buffer length)

		# DEBUG
		self.debug_step1 = 0
		self.debug_step2 = 0

		self.debug_obs1 = []
		self.debug_conv_weight1 = []
		self.debug_decay_weight1 = []
		self.debug_bias1 = []
		self.debug_features1 = []
		self.debug_latent_pi1 = []
		self.debug_latent_vf1 = []
		self.debug_values1 = []
		self.debug_actions1 = []
		self.debug_log_prob1 = []

		self.debug_obs2 = []
		self.debug_conv_weight2 = []
		self.debug_decay_weight2 = []
		self.debug_bias2 = []
		self.debug_features2 = []
		self.debug_latent_pi2 = []
		self.debug_latent_vf2 = []
		self.debug_values2 = []
		self.debug_actions2 = []
		self.debug_log_prob2 = []

	# ----------------------------------------------------------------------------------------------
	def extract_features(self, obs: torch.Tensor, resets: Optional[torch.Tensor] = None, train: bool = False) -> torch.Tensor:
		"""
		Preprocess the observation if needed and extract features.

		:param obs:
		:return:
		"""
		assert self.features_extractor is not None, "No features extractor was set"
		preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
		if isinstance(self.features_extractor, rl.models.EDeNN):
			if train:
				self.prev_features_train = [f.to(self.device) for f in self.prev_features_train]
				features, self.prev_features_train = self.features_extractor(preprocessed_obs, self.prev_features_train)
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
				features, self.prev_features = self.features_extractor(preprocessed_obs, self.prev_features)
				self.prev_features = [f.detach().clone() for f in self.prev_features]
		else:
			features = self.features_extractor(preprocessed_obs)

		return features

	# ----------------------------------------------------------------------------------------------
	def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, resets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
			latent_pi, latent_vf = self.mlp_extractor(features_detached)
		else:
			latent_pi, latent_vf = self.mlp_extractor(features)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)
		entropy = distribution.entropy()

		# DEBUG
		# print(f"{self.debug_step1} {self.debug_step2}: {obs.detach().clone().unique()}")
		self.debug_obs2.append(obs.detach().clone())
		self.debug_features2.append(features.detach().clone())
		# print(f"debug_features2: {self.debug_features2[0].shape} ({len(self.debug_features2)})")
		self.debug_latent_pi2.append(latent_pi.detach().clone())
		self.debug_latent_vf2.append(latent_vf.detach().clone())
		self.debug_values2.append(values.detach().clone())
		self.debug_actions2.append(actions.detach().clone())
		self.debug_log_prob2.append(log_prob.detach().clone())
		self.debug_conv_weight2.append(self.features_extractor.layer1[0].conv_weight.data.detach().clone())
		self.debug_decay_weight2.append(self.features_extractor.layer1[0].decay_weight.data.detach().clone())
		self.debug_bias2.append(self.features_extractor.layer1[0].bias.data.detach().clone())
		self.debug_step2 += 1

		if not self.detach:
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
