import torch
from stable_baselines3.common.policies import ActorCriticPolicy as SB3_ACP

import rl.models

# ==================================================================================================
class ActorCriticPolicy(SB3_ACP):
	def __init__(self, *args, **kwargs):
		"""Subclass `ActorCriticPolicy` to handle gradient breaks for the feature extractor."""
		super().__init__(*args, **kwargs)
		self.layer1_out = None
		self.layer2_out = None
		self.layer3_out = None
		self.layer4_out = None

	# ----------------------------------------------------------------------------------------------
	def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Evaluate actions according to the current policy, given the observations.

		Args:
			obs (torch.Tensor): Observation.
			actions (torch.Tensor): Actions.

		Returns:
			tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: estimated values, log likelihood of taking those actions, entropy of the action distribution, features or projection
		"""
		if isinstance(self.features_extractor, rl.models.EDeNN):
			self.features_extractor.layer1_out = self.layer1_out
			self.features_extractor.layer2_out = self.layer2_out
			self.features_extractor.layer3_out = self.layer3_out
			self.features_extractor.layer4_out = self.layer4_out
			features = self.extract_features(obs)
			if self.features_extractor.projection_head is not None:
				projection = self.features_extractor.project(features)
			self.layer1_out = self.features_extractor.layer1_out
			self.layer2_out = self.features_extractor.layer2_out
			self.layer3_out = self.features_extractor.layer3_out
			self.layer4_out = self.features_extractor.layer4_out
		else:
			features = self.extract_features(obs)

		features_detached = features.clone().detach()
		latent_pi, latent_vf = self.mlp_extractor(features_detached)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)

		# `self.features_extractor` always feeds through its layers up to `layer_last` to produce `features`.
		# If using a projection head, `features_dim` will likely be larger, with `projection_head` being the original size of `features_dim`.
		# In this case, `self.features_extractor.project()` needs to be called manually, same with the loss based on that output.
		# Gradients are detached from the input to the RL policy model, so the `features_extractor` is only trained using this.
		# So `projection` is only used for `bs_loss`.
		if isinstance(self.features_extractor, rl.models.EDeNN) and self.features_extractor.projection_head is not None:
			return values, log_prob, distribution.entropy(), projection
		else:
			return values, log_prob, distribution.entropy(), features
