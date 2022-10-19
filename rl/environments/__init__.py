from .cartpole import CartPoleEvents
from .mountaincar import MountainCarEvents

from gym.envs.registration import register, EnvSpec
# ARBITRARY arguments are passed to EnvSpec

ver = "v0"

register(
	id=f"{CartPoleEvents.__qualname__}-{ver}",
	entry_point=CartPoleEvents,
	max_episode_steps=500, # CartPole-v1
	reward_threshold=475.0, # CartPole-v1
)
register(
	id=f"{MountainCarEvents.__qualname__}-{ver}",
	entry_point=MountainCarEvents,
	max_episode_steps=200, # MountainCar-v0
	reward_threshold=110.0, # MountainCar-v0
)
