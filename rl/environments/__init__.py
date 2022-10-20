from .cartpole import CartPoleEvents
from .mountaincar import MountainCarEvents
from .pong import PongEvents

from gym.envs.registration import register, EnvSpec
# ARBITRARY arguments are passed to EnvSpec

version = "v0"

register(
	id=f"{CartPoleEvents.__qualname__}-{version}",
	entry_point=CartPoleEvents,
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
	id=f"{PongEvents.__qualname__}-{version}",
	entry_point=PongEvents,
	kwargs=dict(
		frameskip=(2, 5), # XXXXXX-v4
		repeat_action_probability=0.0, # XXXXXX-v4
		full_action_space=False, # XXXXXX-v4
		max_num_frames_per_episode=108_000, # XXXXXX-v4
	),
)
