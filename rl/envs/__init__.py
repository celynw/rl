from .cartpole_contrast import CartPoleEnvContrast
from .cartpole_events import CartPoleEnvEvents
from .cartpole_events_debug import CartPoleEnvEventsDebug
from gym.envs.registration import register

register(id="CartPole-contrast-v1", entry_point="rl.envs:CartPoleEnvContrast")
register(id="CartPole-events-v1", entry_point="rl.envs:CartPoleEnvEvents")
register(id="CartPole-events-debug", entry_point="rl.envs:CartPoleEnvEventsDebug", max_episode_steps=10)
