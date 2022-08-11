"""For reinforcement learning."""
from dataclasses import dataclass

import torch

# ==================================================================================================
@dataclass
class Experience:
	"""Experience step gathered in training for off-policy algorithms."""
	state: torch.Tensor
	action: int
	reward: float
	done: bool
	new_state: torch.Tensor


# ==================================================================================================
@dataclass
class Memory: # TODO rename and move to own file (remember to update __init__.py), or consolidate with Experience
	"""Experience step gathered in training for on-policy algorithms."""
	# state: torch.Tensor
	# action: int
	log_prob: float
	value: float
	reward: float
	done: bool
	# new_state: torch.Tensor
