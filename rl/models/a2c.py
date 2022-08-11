import argparse
import inspect
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pytorch_lightning as pl
import numpy as np
import optuna

from rl.utils import Step, Dir
from rl.datasets.utils import RolloutBuffer, RolloutBufferSamples


# From stable baselines
def explained_variance(y_pred: torch.tensor, y_true: torch.tensor) -> np.ndarray:
	"""
	Computes fraction of variance that ypred explains about y.
	Returns 1 - Var[y-ypred] / Var[y]
	interpretation:
		ev=0  =>  might as well have predicted zero
		ev=1  =>  perfect prediction
		ev<0  =>  worse than just predicting zero
	:param y_pred: (np.ndarray) the prediction
	:param y_true: (np.ndarray) the expected value
	:return: (float) explained variance of ypred and y
	"""
	assert y_true.ndim == 1 and y_pred.ndim == 1
	var_y = torch.var(y_true)
	return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y


def maybe_make_env(env, verbose: int):
	"""If env is a string, make the environment; otherwise, return env.

	:param env: (Union[GymEnv, str, None]) The environment to learn from.
	:param monitor_wrapper: (bool) Whether to wrap env in a Monitor when creating env.
	:param verbose: (int) logging verbosity
	:return A Gym (vector) environment.
	"""
	if isinstance(env, str):
		if verbose >= 1:
			print(f"Creating environment from the given name '{env}'")
		env = gym.make(env)

	return env



class BaseModel(pl.LightningModule):
	"""
	The base of RL algorithms

	:param env: The environment to learn from
		(if registered in Gym, can be str. Can be None for loading trained models)
	:param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
		(if registered in Gym, can be str. Can be None for loading trained models)
	:param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
	:param verbose: The verbosity level: 0 none, 1 training information, 2 debug
	:param support_multi_env: Whether the algorithm supports training
		with multiple environments in parallel
	:param seed: Seed for the pseudo random generators
	:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
	"""

	# def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None) -> None:
	def __init__(
		self,
		env,
		eval_env,
		num_eval_episodes: int = 10,
		verbose: int = 0,
		support_multi_env: bool = False,
		seed: Optional[int] = None,
		use_sde: bool = False,
	):
		super().__init__()

		self.num_eval_episodes = num_eval_episodes
		self.verbose = verbose

		# When using VecNormalize:
		self._episode_num = 0
		# Used for gSDE only
		self.use_sde = use_sde

		# Create the env for training and evaluation
		self.env = maybe_make_env(env, self.verbose)
		self.eval_env = maybe_make_env(eval_env, self.verbose)

		# Wrap the env if necessary
		# self.env = wrap_env(self.env, self.verbose)
		# self.eval_env = wrap_env(self.eval_env, self.verbose)

		# print(self.env)
		# from rich import inspect as rinspect
		# rinspect(self.env, all=True)
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		# self.n_envs = self.env.num_envs
		self.n_envs = 1

		if seed:
			self.seed = seed
			self.set_random_seed(self.seed)

		if not support_multi_env and self.n_envs > 1:
			raise ValueError(
				"Error: the model does not support multiple envs; it requires " "a single vectorized environment."
			)

		if self.use_sde and not isinstance(self.action_space, gym.spaces.Box):
			raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

		self.reset()


	def predict(self, obs, deterministic: bool = False) -> np.ndarray:
		"""
		Override this function with the predict function of your own model

		:param obs: The input observations
		:param deterministic: Whether to predict deterministically
		:return: The chosen actions
		"""
		raise NotImplementedError


	def save_hyperparameters(self, frame=None, exclude=['env', 'eval_env']):
		"""
		Utility function to save the hyperparameters of the model.
		This function behaves identically to LightningModule.save_hyperparameters, but will by default exclude the Gym environments
		See https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html#lightningmodule-hyperparameters for more details
		"""
		if not frame:
			frame = inspect.currentframe().f_back
		if not exclude:
			return super().save_hyperparameters(frame=frame)
		if isinstance(exclude, str):
			exclude = (exclude, )
		init_args = pl.utilities.parsing.get_init_args(frame)
		include = [k for k in init_args.keys() if k not in exclude]

		if len(include) > 0:
			super().save_hyperparameters(*include, frame=frame)


	def sample_action(
		self, obs: np.ndarray, deterministic: bool = False
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Samples an action from the environment or from our model

		:param obs: The input observation
		:param deterministic: Whether we are sampling deterministically.
		:return: The action to step with, and the action to store in our buffer
		"""
		with torch.no_grad():
			obs = torch.tensor(obs).to(self.device)
			obs = obs.float() # DEBUG encountered double, should be float
			action = self.predict(obs, deterministic=deterministic)

		if isinstance(self.action_space, gym.spaces.Box):
			action = np.clip(action, self.action_space.low, self.action_space.high)
		elif isinstance(self.action_space, (gym.spaces.Discrete,
											gym.spaces.MultiDiscrete,
											gym.spaces.MultiBinary)):
			action = action.astype(np.int32)
		return action


	def evaluate(
		self,
		num_eval_episodes: int,
		deterministic: bool = True,
		render: bool = False,
		# render: bool = True,
		record: bool = False,
		record_fn: Optional[str] = None) -> Tuple[List[float], List[int]]:
		"""
		Evaluate the model with eval_env

		:param num_eval_episodes: Number of episodes to evaluate for
		:param deterministic: Whether to evaluate deterministically
		:param render: Whether to render while evaluating
		:param record: Whether to recod while evaluating
		:param record_fn: File to record environment to if we are recording
		:return: A list of total episode rewards and a list of episode lengths
		"""

		# if isinstance(self.eval_env, VecEnv):
		# 	assert self.eval_env.num_envs == 1, "Cannot run eval_env in parallel. eval_env.num_env must equal 1"

		# if not is_wrapped(self.eval_env, Monitor) and self.verbose:
		# 	warnings.warn(
		# 		"Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
		# 		"This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
		# 		"Consider wrapping environment first with ``Monitor`` wrapper.",
		# 		UserWarning,
		# 	)

		episode_rewards, episode_lengths = [], []

		if record:
			recorder = VideoRecorder(env=self.eval_env, path=record_fn)

		not_reseted = True
		for i in range(num_eval_episodes):
			done = False
			episode_rewards += [0.0]
			episode_lengths += [0]

			# Number of loops here might differ from true episodes
			# played, if underlying wrappers modify episode lengths.
			# Avoid double reset, as VecEnv are reset automatically.
			# if not isinstance(self.eval_env, VecEnv) or not_reseted:
			if not_reseted:
				obs = self.eval_env.reset()
				not_reseted = False

			while not done:
				action = self.sample_action(obs, deterministic)

				action = action[0] # DEBUG it's a 1 element list or numpy array, just need the element
				obs, reward, done, info = self.eval_env.step(action)
				episode_rewards[-1] += reward
				episode_lengths[-1] += 1

				if render:
					self.eval_env.render()
				if record:
					recorder.capture_frame()

			# if is_wrapped(self.eval_env, Monitor):
			# 	# Do not trust "done" with episode endings.
			# 	# Remove vecenv stacking (if any)
			# 	# if isinstance(self.eval_env, VecEnv):
			# 	# 	info = info[0]
			# 	if "episode" in info.keys():
			# 		# Monitor wrapper includes "episode" key in info if environment
			# 		# has been wrapped with it. Use those rewards instead.
			# 		episode_rewards[-1] = info["episode"]["r"]
			# 		episode_lengths[-1] = info["episode"]["l"]
		if record:
			recorder.close()

		return episode_rewards, episode_lengths


	def training_epoch_end(self, outputs) -> None:
		"""
		Run the evaluation function at the end of the training epoch
		Override this if you also wish to do other things at the end of a training epoch
		"""
		self.eval()
		rewards, lengths = self.evaluate(self.num_eval_episodes)
		self.train()
		self.log_dict({
			'val_reward_mean': np.mean(rewards),
			'val_reward_std': np.std(rewards),
			'val_lengths_mean': np.mean(lengths),
			'val_lengths_std': np.std(lengths)},
			prog_bar=True, logger=True)


	def reset(self) -> None:
		"""
		Reset the enviornment
		"""
		self._last_obs = self.env.reset()
		# self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)
		self._last_dones = np.zeros((1,), dtype=np.bool)


	def set_random_seed(self, seed: int) -> None:
		"""
		Set the seed of the pseudo-random generators
		(python, numpy, pytorch, gym)

		:param seed: The random seed to set
		"""
		set_random_seed(seed)
		self.action_space.seed(seed)
		if self.env:
			self.env.seed(seed)
		if self.eval_env:
			self.eval_env.seed(seed)



class OnPolicyModel(BaseModel):
	"""
	The base for On-Policy algorithms (ex: A2C/PPO).

	:param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
	:param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
		(if registered in Gym, can be str. Can be None for loading trained models)
	:param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
	:param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
		just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
	:param batch_size: Minibatch size for each gradient update
	:param epochs_per_rollout: Number of epochs to optimise the loss for
	:param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
	:param gamma: (float) Discount factor
	:param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
		Equivalent to classic advantage when set to 1.
	:param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration
	:param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param verbose: The verbosity level: 0 none, 1 training information, 2 debug
	:param seed: Seed for the pseudo random generators
	"""

	def __init__(
		self,
		env,
		eval_env,
		buffer_length: int,
		num_rollouts: int,
		batch_size: int,
		epochs_per_rollout: int,
		num_eval_episodes: int = 10,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		verbose: int = 0,
		seed: Optional[int] = None,
	):
		super().__init__(
			env=env,
			eval_env=eval_env,
			num_eval_episodes=num_eval_episodes,
			verbose=verbose,
			support_multi_env=True,
			seed=seed,
			use_sde=use_sde,
		)

		self.buffer_length = buffer_length
		self.num_rollouts = num_rollouts
		self.batch_size = batch_size
		self.epochs_per_rollout = epochs_per_rollout
		self.gamma = gamma
		self.gae_lambda = gae_lambda

		self.rollout_buffer = RolloutBuffer(
			buffer_length,
			self.observation_space,
			self.action_space,
			gamma=self.gamma,
			gae_lambda=self.gae_lambda,
			n_envs=self.n_envs,
		)


	def forward(self, obs) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
		"""
		Override this function with the forward function of your model

		:param obs: The input observations
		:return: The chosen actions
		"""
		raise NotImplementedError


	def train_dataloader(self):
		"""
		Create the dataloader for our OffPolicyModel
		"""
		return OnPolicyDataloader(self)


	def collect_rollouts(self) -> RolloutBufferSamples:
		"""
		Collect rollouts and put them into the RolloutBuffer
		"""
		assert self._last_obs is not None, "No previous observation was provided"
		with torch.no_grad():
			# Sample new weights for the state dependent exploration
			if self.use_sde:
				self.reset_noise(self.env.num_envs)

			self.eval()
			for i in range(self.buffer_length):
				if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
					# Sample a new noise matrix
					self.reset_noise(self.env.num_envs)

				# Convert to pytorch tensor, let Lightning take care of any GPU transfer
				obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
				dist, values = self(obs_tensor)
				actions = dist.sample()
				log_probs = dist.log_prob(actions)

				# Rescale and perform action
				clipped_actions = actions.cpu().numpy()
				# Clip the actions to avoid out of bound error
				if isinstance(self.action_space, gym.spaces.Box):
					clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
				elif isinstance(self.action_space, gym.spaces.Discrete):
					clipped_actions = clipped_actions.astype(np.int32)

				clipped_actions = clipped_actions[0] # DEBUG it's a 1 element list or numpy array, just need the element
				if self._last_dones:
					print("STEPPING WITH DONES:", self._last_dones)
				new_obs, rewards, dones, infos = self.env.step(clipped_actions)

				if isinstance(self.action_space, gym.spaces.Discrete):
					# Reshape in case of discrete action
					actions = actions.view(-1, 1)

				if not torch.is_tensor(self._last_dones):
					self._last_dones = torch.as_tensor(self._last_dones).to(device=obs_tensor.device)
				rewards = torch.as_tensor(rewards).to(device=obs_tensor.device)

				self.rollout_buffer.add(obs_tensor, actions, rewards, self._last_dones, values, log_probs)
				self._last_obs = new_obs
				self._last_dones = dones

				# DEBUG
				if dones:
					self.reset()

			final_obs = torch.as_tensor(new_obs).to(device=self.device, dtype=torch.float32)
			dist, final_values = self(final_obs)
			samples = self.rollout_buffer.finalize(final_values, torch.as_tensor(dones).to(device=obs_tensor.device, dtype=torch.float32))

			self.rollout_buffer.reset()
		self.train()
		return samples



class OnPolicyDataloader:
	def __init__(self, model: OnPolicyModel):
		self.model = model


	def __iter__(self):
		for i in range(self.model.num_rollouts):
			experiences = self.model.collect_rollouts()
			observations, actions, old_values, old_log_probs, advantages, returns = experiences
			for j in range(self.model.epochs_per_rollout):
				k = 0
				perm = torch.randperm(observations.shape[0], device=observations.device)
				while k < observations.shape[0]:
					batch_size = min(observations.shape[0] - k, self.model.batch_size)
					yield RolloutBufferSamples(
						observations[perm[k:k+batch_size]],
						actions[perm[k:k+batch_size]],
						old_values[perm[k:k+batch_size]],
						old_log_probs[perm[k:k+batch_size]],
						advantages[perm[k:k+batch_size]],
						returns[perm[k:k+batch_size]])
					k += batch_size



class A2C__(OnPolicyModel):
	"""
	Advantage Actor Critic (A2C)

	Paper: https://arxiv.org/abs/1602.01783
	Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
	and Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

	Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

	:param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
	:param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
		(if registered in Gym, can be str. Can be None for loading trained models)
	:param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
	:param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
		just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
	:param batch_size: Minibatch size for each gradient update
	:param epochs_per_rollout: Number of epochs to optimise the loss for
	:param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
	:param gamma: (float) Discount factor
	:param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
		Equivalent to classic advantage when set to 1.
	:param value_coef: Value function coefficient for the loss calculation
	:param entropy_coef: Entropy coefficient for the loss calculation
	:param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration
	:param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param verbose: The verbosity level: 0 none, 1 training information, 2 debug
	:param seed: Seed for the pseudo random generators
	"""
	def __init__(
		self,
		env,
		eval_env,
		buffer_length: int = 5,
		num_rollouts: int = 100,
		batch_size: int = 128,
		epochs_per_rollout: int = 1,
		num_eval_episodes: int = 10,
		gamma: float = 0.99,
		gae_lambda: float = 1.0,
		value_coef: float = 0.5,
		entropy_coef: float = 0.0,
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		verbose: int = 0,
		seed: Optional[int] = None,
	):
		super().__init__(
			env=env,
			eval_env=eval_env,
			buffer_length=buffer_length,
			num_rollouts=num_rollouts,
			batch_size=batch_size,
			epochs_per_rollout=epochs_per_rollout,
			num_eval_episodes=num_eval_episodes,
			gamma=gamma,
			gae_lambda=gae_lambda,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			verbose=verbose,
			seed=seed
		)

		self.value_coef = value_coef
		self.entropy_coef = entropy_coef

	def forward(
		self, x: torch.Tensor
		) -> Tuple[distributions.Distribution, torch.Tensor]:
		"""
		Runs both the actor and critic network

		:param x: The input observations
		:return: The deterministic action of the actor
		"""
		raise NotImplementedError

	def training_step(self, batch, batch_idx):
		"""
		Specifies the update step for A2C. Override this if you wish to modify the A2C algorithm
		"""
		if self.use_sde:
			self.reset_noise(self.batch_size)

		dist, values = self(batch.observations)
		log_probs = dist.log_prob(batch.actions)
		values = values.flatten()

		advantages = batch.advantages.detach()
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		policy_loss = -(advantages * log_probs).mean()
		value_loss = F.mse_loss(batch.returns.detach(), values)
		entropy_loss = -dist.entropy().mean()

		loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

		with torch.no_grad():
			explained_var = explained_variance(batch.old_values, batch.returns)
		self.log_dict({
			'train_loss': loss,
			'policy_loss': policy_loss,
			'value_loss': value_loss,
			'entropy_loss': entropy_loss,
			'explained_var': explained_var},
			prog_bar=False, logger=True)

		return loss

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Add model-specific arguments.

		Args:
			parser (argparse.ArgumentParser): Main parser

		Returns:
			argparse.ArgumentParser: Modified parser
		"""
		group = parser.add_argument_group("Model")
		envs = [env_spec.id for env_spec in gym.envs.registry.all()]
		group.add_argument("env", choices=envs, metavar=f"ENV", help="AI gym environment to use")
		args_known, _ = parser.parse_known_args()
		if args_known.env is None:
			parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

		group.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
		group.add_argument("--epoch_length", type=int, default=200, help="How many experiences to sample per pytorch epoch")
		# group.add_argument("--replay_size", type=int, default=1000, help="Capacity of the replay buffer")
		# group.add_argument("--warm_start_steps", type=int, default=1000, help="Number of iterations for linear warmup") # TODO pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
		group.add_argument("--sync_rate", type=int, default=10, help="How many frames used to update the target network")
		# TODO refactor EPS here
		# group.add_argument("--eps_start", type=float, default=1.0, help="Epsilon starting value")
		# group.add_argument("--eps_end", type=float, default=0.01, help="Epsilon final value")
		# group.add_argument("--eps_last_frame", type=int, default=1000, help="Which frame epsilon should stop decaying at")

		# DEBUG
		# parser = RLOnPolicy.add_argparse_args(parser)
		group = parser.add_argument_group("Dataset")
		group.add_argument("-b", "--batch_size", type=int, help="Batch size", default=16)
		group.add_argument("-a", "--batch_accumulation", type=int, help="Perform batch accumulation", default=1)
		group.add_argument("-w", "--workers", type=int, help="Dataset workers, can use 0", default=0)

		return parser



class A2C_(A2C__):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if isinstance(self.action_space, gym.spaces.Discrete):
			self.p = nn.Parameter(torch.ones(1, self.action_space.n) * 0.5)
		elif isinstance(self.action_space, gym.spaces.Box):
			self.p = nn.Parameter(torch.ones(1, self.action_space.shape[0] * 2) * 0.5)
		else:
			raise Exception('Incompatible environment action space')

		self.save_hyperparameters()


	def forward(self, x, **kwargs):
		p = self.p.expand(x.shape[0], self.p.shape[-1])
		if isinstance(self.action_space, gym.spaces.Discrete):
			dist = distributions.Categorical(probs=F.softmax(p, dim=1))
		elif isinstance(self.action_space, gym.spaces.Box):
			p = torch.chunk(p, 2, dim=1)
			dist = distributions.Normal(loc=p[0], scale=p[1])
		return dist, torch.ones_like(x)[:, :1]


	def predict(self, x, deterministic=True):
		p = self.p.expand(x.shape[0], self.p.shape[-1])
		if deterministic:
			if isinstance(self.action_space, gym.spaces.Discrete):
				out = torch.max(p, dim=1)[1]
			elif isinstance(self.action_space, gym.spaces.Box):
				out = torch.chunk(p, 2, dim=1)[0]
		else:
			if isinstance(self.action_space, gym.spaces.Discrete):
				out = distributions.Categorical(probs=F.softmax(p, dim=1)).sample()
			elif isinstance(self.action_space, gym.spaces.Box):
				p = torch.chunk(p, 2, dim=1)
				out = distributions.Normal(loc=p[0], scale=p[1]).sample()
		return out.cpu().numpy()


	def configure_optimizers(self):
		return torch.optim.Adam(params=self.parameters(), lr=1e-3)


class A2C(A2C_):
	# monitor = f"{Step.TRAIN}/mean_reward" # TODO
	monitor = "val_reward_mean" # TODO
	monitor_dir = Dir.MAX

	# def __init__(self, *args, **kwargs):
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None) -> None:
	# def __init__(
	# 	self,
	# 	env,
	# 	eval_env,
	# 	num_eval_episodes: int = 10,
	# 	verbose: int = 0,
	# 	support_multi_env: bool = False,
	# 	seed: Optional[int] = None,
	# 	use_sde: bool = False,
	# ):
		# super().__init__(*args, **kwargs)
		super().__init__(
			env=gym.make(args.env),
			eval_env=gym.make(args.env),
			num_eval_episodes=10,
			verbose=9,
			use_sde=False,
			seed=None,
			# support_multi_env=False,
		)

		self.actor = nn.Sequential(
			nn.Linear(self.observation_space.shape[0], 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, self.action_space.n),
		# )
			nn.Softmax(dim=1))
		# self.sm = nn.Softmax(dim=1)

		self.critic = nn.Sequential(
			nn.Linear(self.observation_space.shape[0], 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1))

		self.save_hyperparameters()

	# This is for training the model
	# Returns the distribution and the corresponding value
	def forward(self, x):
		# if x.dim() == 1:
		x = x[None, ...] # DEBUG it needs to be by batch
		# print(f"x.shape: {x.shape}")
		out = self.actor(x)
		# print(f"out.shape: {out.shape}")
		# out = self.sm(out)
		dist = distributions.Categorical(probs=out)
		return dist, self.critic(x).flatten()

	# This is for inference and evaluation of our model, returns the action
	def predict(self, x, deterministic=True):
		x = x[None, ...] # DEBUG it needs to be by batch
		# print(f"PRED x.shape: {x.shape}")
		out = self.actor(x)
		# print(f"PRED out.shape: {out.shape}")
		# out = self.sm(out)
		if deterministic:
			out = torch.max(out, dim=1)[1]
		else:
			out = distributions.Categorical(probs=out).sample()
		return out.cpu().numpy()

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
		return optimizer
