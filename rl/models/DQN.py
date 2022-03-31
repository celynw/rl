#!/usr/bin/env python3
import torch

from stable_baselines3 import DQN as DQN_sb3

from rl.utils import Step
from rl.models import RL

# ==================================================================================================
class DQN(RL):
	"""Deep Q Network."""
	env_type = "CartPole-v0"
	sb3_model = DQN_sb3

	# ----------------------------------------------------------------------------------------------
	def step(self, step: Step, batch, batch_idx: int):
		return self(batch)

	# ----------------------------------------------------------------------------------------------
	def get_layers(self):
		# TODO separate actor and critic
		# TODO input shape: self.observation_space.shape[0]
		# TODO output shape: self.action_space.n
		self.l1 = torch.nn.Linear(8, 64)
		self.l2 = torch.nn.Linear(64, 64)
		self.l3 = torch.nn.Linear(64, 3)

		return torch.nn.Sequential(
			self.l1,
			torch.nn.Tanh(),
			self.l2,
			torch.nn.Tanh(),
			self.l3,
			torch.nn.Softmax(dim=1)
		)


# ==================================================================================================
if __name__ == "__main__":
	import argparse
	# from rl2.utils import parse_args
	from stable_baselines3 import DQN as DQN_sb3

	# args = parse_args()
	parser = argparse.ArgumentParser()
	model = DQN(args=parser.parse_args(), trial=None)

	model = DQN_sb3("MlpPolicy", model.train_env, verbose=1)
	model.learn(total_timesteps=10000, log_interval=4)
	model.save("dqn_cartpole")

	del model # remove to demonstrate saving and loading

	model = DQN_sb3.load("dqn_cartpole")

	obs = env.reset()
	while True:
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			obs = env.reset()
