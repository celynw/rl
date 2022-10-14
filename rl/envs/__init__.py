from .cartpole_events import CartPoleEnvEvents
from .cartpole_events_sim import CartPoleEnvEventsSim
from .cartpole_events_debug import CartPoleEnvEventsDebug
from .cartpole_rgb import CartPoleEnvRGB
from .mountaincar_events import MountainCarEnvEvents
from .mountaincar_rgb import MountainCarEnvRGB
from .atari_events import AtariEnvEvents
from gym.envs.registration import register

# NOTE: reward_threshold isn't actually used, it's just available metadata

register(
	id="CartPole-events-v1",
	entry_point="rl.envs:CartPoleEnvEvents",
	max_episode_steps=500,
	reward_threshold=475.0,
)
register(
	id="CartPole-events-sim-v1",
	entry_point="rl.envs:CartPoleEnvEventsSim",
	max_episode_steps=500,
	reward_threshold=475.0,
)
register(
	id="CartPole-events-debug",
	entry_point="rl.envs:CartPoleEnvEventsDebug",
	max_episode_steps=10,
	reward_threshold=475.0,
)
register(
	id="CartPole-rgb",
	entry_point="rl.envs:CartPoleEnvRGB",
	max_episode_steps=500,
	reward_threshold=475.0,
)
register(
	id="MountainCar-events-v0",
	entry_point="rl.envs:MountainCarEnvEvents",
	max_episode_steps=200,
	reward_threshold=-110.0,
)
register(
	id="MountainCar-rgb-v0",
	entry_point="rl.envs:MountainCarEnvRGB",
	max_episode_steps=200,
	reward_threshold=-110.0,
)
register(
	id="Pong-events-v0",
	entry_point="rl.envs:AtariEnvEvents",
)
