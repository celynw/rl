from .cartpole_contrast import CartPoleEnvContrast
from .cartpole_events import CartPoleEnvEvents
from gym.envs.registration import register

register(id="CartPole-contrast-v1", entry_point="rl.envs:CartPoleEnvContrast")
register(id="CartPole-events-v1", entry_point="rl.envs:CartPoleEnvEvents")
