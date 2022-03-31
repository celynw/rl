"""For reinforcement learning."""
from dataclasses import dataclass

import torch

# ==================================================================================================
@dataclass
class Experience:
	"""For storing experience steps gathered in training."""
	state: torch.Tensor
	action: int
	reward: float
	done: bool
	new_state: torch.Tensor
