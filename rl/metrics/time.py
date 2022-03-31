from typing import Union

import torch
from torchmetrics import Metric

# ==================================================================================================
class InferenceTime(Metric):
	"""Abusing torchmetrics slightly, just used for average step time."""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		# Call `self.add_state` for every internal state that is needed for the metrics computations
		# dist_reduce_fx indicates the function that should be used to reduce state from multiple processes
		self.add_state("time_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.add_state("iters", default=torch.tensor(0), dist_reduce_fx="sum")

	# ----------------------------------------------------------------------------------------------
	def update(self, time: Union[float, torch.Tensor]):
		"""Update metric states."""
		self.iters += 1
		if isinstance(time, (float, int)):
			time = torch.tensor(time, dtype=torch.float)
		self.time_sum += time

	# ----------------------------------------------------------------------------------------------
	def compute(self):
		"""Compute final result."""
		if self.iters == 0:
			print(f"  time: {self.time_sum} / {self.iters} = {self.time_sum / self.iters}")
		return self.time_sum / self.iters
