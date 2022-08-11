#!/usr/bin/env python3
"""https://github.com/hermesdt/reinforcement-learning/blob/master/ppo/cartpole_ppo_online.ipynb"""
import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from tqdm import tqdm

# ==================================================================================================
def mish(input):
	return input * torch.tanh(F.softplus(input))


# ==================================================================================================
class Mish(nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		super().__init__()

	# ----------------------------------------------------------------------------------------------
	def forward(self, input):
		return mish(input)


# ==================================================================================================
# helper function to convert numpy arrays to tensors
def t(x):
	return torch.from_numpy(x).float()


# ==================================================================================================
# Actor module, categorical actions only
class Actor(nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, state_dim, n_actions, activation=nn.Tanh):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(state_dim, 64),
			activation(),
			nn.Linear(64, 32),
			activation(),
			nn.Linear(32, n_actions),
			nn.Softmax()
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, X):
		return self.model(X)


# ==================================================================================================
# Critic module
class Critic(nn.Module):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, state_dim, activation=nn.Tanh):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(state_dim, 64),
			activation(),
			nn.Linear(64, 32),
			activation(),
			nn.Linear(32, 1)
		)

	# ----------------------------------------------------------------------------------------------
	def forward(self, X):
		return self.model(X)


# ==================================================================================================
def clip_grad_norm_(module, max_grad_norm):
	nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


# ==================================================================================================
def policy_loss(old_log_prob, log_prob, advantage, eps):
	ratio = (log_prob - old_log_prob).exp()
	clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage

	m = torch.min(ratio*advantage, clipped)
	return -m


# ==================================================================================================
if __name__ == "__main__":
	env = gym.make("CartPole-v1")
	# config
	state_dim = env.observation_space.shape[0]
	n_actions = env.action_space.n
	all_rewards = []
	for _ in tqdm(range(20)):
		actor = Actor(state_dim, n_actions, activation=Mish)
		critic = Critic(state_dim, activation=Mish)
		adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
		adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

		torch.manual_seed(1)

		episode_rewards = []
		gamma = 0.98
		eps = 0.2
		# w = tensorboard.SummaryWriter()
		s = 0
		max_grad_norm = 0.5

		for i in range(800):
			prev_prob_act = None
			done = False
			total_reward = 0
			state = env.reset()

			while not done:
				s += 1
				probs = actor(t(state))
				dist = torch.distributions.Categorical(probs=probs)
				action = dist.sample()
				prob_act = dist.log_prob(action)

				next_state, reward, done, info = env.step(action.detach().data.numpy())
				advantage = reward + (1-done)*gamma*critic(t(next_state)) - critic(t(state))

				# w.add_scalar("loss/advantage", advantage, global_step=s)
				# w.add_scalar("actions/action_0_prob", dist.probs[0], global_step=s)
				# w.add_scalar("actions/action_1_prob", dist.probs[1], global_step=s)

				total_reward += reward
				state = next_state

				if prev_prob_act:
					actor_loss = policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps)
					# w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
					adam_actor.zero_grad()
					actor_loss.backward()
					# clip_grad_norm_(adam_actor, max_grad_norm)
					# w.add_histogram("gradients/actor", torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=s)
					adam_actor.step()

					critic_loss = advantage.pow(2).mean()
					# w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
					adam_critic.zero_grad()
					critic_loss.backward()
					# clip_grad_norm_(adam_critic, max_grad_norm)
					# w.add_histogram("gradients/critic", torch.cat([p.data.view(-1) for p in critic.parameters()]), global_step=s)
					adam_critic.step()

				prev_prob_act = prob_act

			# w.add_scalar("reward/episode_reward", total_reward, global_step=i)
			episode_rewards.append(total_reward)
		all_rewards.append(episode_rewards)

	all_rewards_ = -np.ones([len(all_rewards), max([len(x) for x in all_rewards])])
	for i, sub in enumerate(all_rewards):
		all_rewards_[i][0:len(sub)] = sub
	all_rewards = np.ma.MaskedArray(all_rewards_, mask=all_rewards_ < 0)

	mean = all_rewards.mean(axis=0)
	std = all_rewards.std(axis=0)
	import matplotlib
	# xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.
	matplotlib.use("agg") # Some problem with "xcb", could be due to not having a screen (headless) over SSH?
	ax = plt.gca()
	print("Done")
	ax.set_ylim([0, 500]) # TODO get env max reward instead of hard coding
	ax.fill_between(
		range(all_rewards.shape[1]), # X
		# np.clip(mean - std, 0, None), # Max
		mean - std, # Max
		mean + std, # Min
		alpha=.5,
		linewidth=0
	)
	ax.plot(mean, linewidth=2)
	plt.savefig(f"mine.png")
