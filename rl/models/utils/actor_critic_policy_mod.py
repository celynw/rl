import torch
from stable_baselines3.common.policies import ActorCriticPolicy

# from rl.models import EDeNN

# ==================================================================================================
class ActorCriticPolicy_mod(ActorCriticPolicy):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.layer1_out = None
		self.layer2_out = None
		self.layer3_out = None
		self.layer4_out = None

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
		# if isinstance(self.features_extractor, "rl.models.EDeNN"): # Forward reference
		if "rl.models.EDeNN" in str(type(self.features_extractor)): # Nasty workaround!
		# try: # FIX can't import EDeNN due to partially initialised module
			# EDeNN (propagate decays between steps)
			assert(hasattr(self.features_extractor, "layer1_out")) # Catch using the wrong model...
			self.features_extractor.layer1_out = self.layer1_out
			self.features_extractor.layer2_out = self.layer2_out
			self.features_extractor.layer3_out = self.layer3_out
			self.features_extractor.layer4_out = self.layer4_out
			features = self.extract_features(obs)
			self.layer1_out = self.features_extractor.layer1_out
			self.layer2_out = self.features_extractor.layer2_out
			self.layer3_out = self.features_extractor.layer3_out
			self.layer4_out = self.features_extractor.layer4_out
		# except AttributeError:
		else:
			features = self.extract_features(obs)
			projectionHead = hasattr(self.features_extractor, "forward_final")
			if projectionHead:
				projection = self.features_extractor.forward_final(features)

		features_detached = features.clone().detach()
		latent_pi, latent_vf = self.mlp_extractor(features_detached)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)

		if projectionHead:
			return values, log_prob, distribution.entropy(), projection
		else:
			return values, log_prob, distribution.entropy(), features
