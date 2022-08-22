"""
Based on CartPole-v1 from gym=0.23.1
Changed colours to increase contrast.

Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from rl.envs.cartpole_events import CartPoleEnvEvents

# ==================================================================================================
class CartPoleEnvEventsSim(CartPoleEnvEvents):
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		super().__init__(init_ros=False)
		self.events = []
