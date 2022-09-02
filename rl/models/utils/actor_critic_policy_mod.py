import torch
from stable_baselines3.common.policies import ActorCriticPolicy

# from rl.models import EDeNN

# ==================================================================================================
class ActorCriticPolicy_mod(ActorCriticPolicy):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.out_c1 = None
		self.out_c2 = None
		self.out_c3 = None
		self.out_c4 = None
		self.out_mid = None

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
			self.features_extractor.out_c1 = self.out_c1
			self.features_extractor.out_c2 = self.out_c2
			self.features_extractor.out_c3 = self.out_c3
			self.features_extractor.out_c4 = self.out_c4
			self.features_extractor.out_mid = self.out_mid
			features = self.extract_features(obs)
			self.out_c1 = self.features_extractor.out_c1
			self.out_c2 = self.features_extractor.out_c2
			self.out_c3 = self.features_extractor.out_c3
			self.out_c4 = self.features_extractor.out_c4
			self.out_mid = self.features_extractor.out_mid
		# except AttributeError:
		else:
			features = self.extract_features(obs)
		features_detached = features.clone().detach()
		latent_pi, latent_vf = self.mlp_extractor(features_detached)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)

		return values, log_prob, distribution.entropy(), features
