import torch
from stable_baselines3.common.policies import ActorCriticPolicy

# ==================================================================================================
class ActorCriticPolicy_mod(ActorCriticPolicy):
	# ----------------------------------------------------------------------------------------------
	def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Evaluate actions according to the current policy,
		given the observations.

		:param obs:
		:param actions:
		:return: estimated value, log likelihood of taking those actions
			and entropy of the action distribution.
		"""
		# Preprocess the observation if needed
		features = self.extract_features(obs)
		features_detached = features.clone().detach()
		latent_pi, latent_vf = self.mlp_extractor(features_detached)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)

		return values, log_prob, distribution.entropy(), features
