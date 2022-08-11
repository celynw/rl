from .cartpole_contrast import CartPoleEnvContrast
from gym.envs.registration import register

register(id="CartPole-contrast-v1", entry_point="rl.envs:CartPoleEnvContrast")
