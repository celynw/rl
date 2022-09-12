import torch

# ==================================================================================================
class SpikeRepresentationGenerator:
	"""Generate spikes from event tensors."""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, height: int, width: int, num_time_bins: int):
		"""
		Generate spikes from event tensors.
		Args:
			height (int): Height of event image
			width (int): Width of event image
		"""
		self.height = height
		self.width = width
		self.num_time_bins = num_time_bins

	# ----------------------------------------------------------------------------------------------
	def getSlayerSpikeTensor(self, ev_pol: torch.Tensor, ev_xy: torch.Tensor,
				ev_ts_us: torch.Tensor) -> torch.Tensor:
		"""
		Generate spikes from event tensors.
		All arguments must be of the same image shape.
		Args:
			ev_pol (torch.Tensor): Event polarities
			ev_xy (torch.Tensor): Event locations
			ev_ts_us (torch.Tensor): Event timestamps in microseconds
		Returns:
			torch.Tensor: Spike train tensor
		"""
		spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))
		if len(ev_ts_us) < 2:
			return spike_tensor # Empty tensor, don't raise an error

		binDuration = (ev_ts_us[-1] - ev_ts_us[0]) / self.num_time_bins
		if binDuration == 0:
			return spike_tensor
		# print(f"binDuration: {binDuration}")
		time_idx = ((ev_ts_us - ev_ts_us[0]) / binDuration)
		# print(f"ev_ts_us[0]: {ev_ts_us[0]}")
		# print(f"ev_ts_us: {ev_ts_us}")
		# print(f"(ev_ts_us - ev_ts_us[0]): {(ev_ts_us - ev_ts_us[0])}")
		# print(f"time_idx: {time_idx}")
		# Handle time stamps that are not floored and would exceed the allowed index after to-index conversion.
		time_idx[time_idx >= self.num_time_bins] = self.num_time_bins - 1

		spike_tensor[ev_pol.long(), ev_xy[:, 1].long(), ev_xy[:, 0].long(), time_idx.long()] = 1

		return spike_tensor
