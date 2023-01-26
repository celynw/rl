from .cartpole import CartPoleEvents, CartPoleRGB
from .pendulum import PendulumEvents
from .mountaincar import MountainCarEvents, MountainCarRGB
from .pong import PongEvents, PongRGB
from .freeway import FreewayEvents
from .skiing import SkiingEvents

from gymnasium.envs.registration import register#, EnvSpec
# ARBITRARY arguments are passed to EnvSpec

version = "v0"

register(
	id=f"{CartPoleEvents.__qualname__}-{version}",
	entry_point=CartPoleEvents,
	max_episode_steps=500, # CartPole-v1
	reward_threshold=475.0, # CartPole-v1
)
register(
	id=f"{PendulumEvents.__qualname__}-{version}",
	entry_point=PendulumEvents,
	max_episode_steps=200, # Pendulum-v1
	reward_threshold=0, # Pendulum-v1
)
register(
	id=f"{CartPoleRGB.__qualname__}-{version}",
	entry_point=CartPoleRGB,
	max_episode_steps=500, # CartPole-v1
	reward_threshold=475.0, # CartPole-v1
)
register(
	id=f"{MountainCarEvents.__qualname__}-{version}",
	entry_point=MountainCarEvents,
	max_episode_steps=200, # MountainCar-v0
	reward_threshold=110.0, # MountainCar-v0
)
register(
	id=f"{MountainCarRGB.__qualname__}-{version}",
	entry_point=MountainCarRGB,
	max_episode_steps=200, # MountainCar-v0
	reward_threshold=110.0, # MountainCar-v0
)
register(
	id=f"{PongEvents.__qualname__}-{version}",
	entry_point=PongEvents,
	kwargs=dict(
		# frameskip=(2, 5), # XXXXXX-v4
		frameskip=1,
		repeat_action_probability=0.0, # XXXXXX-v4
		full_action_space=False, # XXXXXX-v4
		max_num_frames_per_episode=108_000, # XXXXXX-v4
	),
)
register(
	id=f"{PongRGB.__qualname__}-{version}",
	entry_point=PongRGB,
	kwargs=dict(
		# frameskip=(2, 5), # XXXXXX-v4
		frameskip=1, # XXXXXXNoFrameSkip-v4
		repeat_action_probability=0.0, # XXXXXX-v4
		full_action_space=False, # XXXXXX-v4
		max_num_frames_per_episode=108_000, # XXXXXX-v4
	),
)
register(
	id=f"{FreewayEvents.__qualname__}-{version}",
	entry_point=FreewayEvents,
	kwargs=dict(
		frameskip=(2, 5), # XXXXXX-v4
		repeat_action_probability=0.0, # XXXXXX-v4
		full_action_space=False, # XXXXXX-v4
		max_num_frames_per_episode=108_000, # XXXXXX-v4
	),
)
register(
	id=f"{SkiingEvents.__qualname__}-{version}",
	entry_point=SkiingEvents,
	kwargs=dict(
		frameskip=(2, 5), # XXXXXX-v4
		repeat_action_probability=0.0, # XXXXXX-v4
		full_action_space=False, # XXXXXX-v4
		max_num_frames_per_episode=108_000, # XXXXXX-v4
	),
)
