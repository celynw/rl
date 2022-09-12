from .cartpole_events import CartPoleEnvEvents
from .cartpole_events_sim import CartPoleEnvEventsSim
from .cartpole_events_debug import CartPoleEnvEventsDebug
from .cartpole_rgb import CartPoleEnvRGB
from .mountaincar_events import MountainCarEnvEvents
from gym.envs.registration import register

register(id="CartPole-events-v1", entry_point="rl.envs:CartPoleEnvEvents")
register(id="CartPole-events-sim-v1", entry_point="rl.envs:CartPoleEnvEventsSim")
register(id="CartPole-events-debug", entry_point="rl.envs:CartPoleEnvEventsDebug", max_episode_steps=10)
register(id="CartPole-rgb", entry_point="rl.envs:CartPoleEnvRGB")
register(id="MountainCar-events-v0", entry_point="rl.envs:MountainCarEnvEvents")
