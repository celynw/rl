import torch

# ==================================================================================================
class SpikeRepresentationGenerator:
	# ----------------------------------------------------------------------------------------------
	def __init__(self, height: int, width: int, num_time_bins: int):
		"""
		Class to generate spikes from event tensors.
		Args:
			height (int): Height of event image
			width (int): Width of event image
		"""
		self.height = height
		self.width = width
		self.num_time_bins = num_time_bins

	# ----------------------------------------------------------------------------------------------
	def getSlayerSpikeTensor(self, polarities: torch.Tensor, locations: torch.Tensor,
			timestamps: torch.Tensor) -> torch.Tensor:
		"""
		Generate spikes from event tensors.
		All arguments must be of the same image shape.

		Args:
			polarities (torch.Tensor): Event polarities
			locations (torch.Tensor): Event XY locations
			timestamps (torch.Tensor): Event timestamps in microseconds

		Returns:
			torch.Tensor: Spike train tensor
		"""
		spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))
		if len(timestamps) < 2:
			return spike_tensor # Empty tensor, don't raise an error

		binDuration = (timestamps[-1] - timestamps[0]) / self.num_time_bins
		if binDuration == 0:
			return spike_tensor
		# print(f"binDuration: {binDuration}")
		time_idx = ((timestamps - timestamps[0]) / binDuration)
		# print(f"timestamps[0]: {timestamps[0]}")
		# print(f"timestamps: {timestamps}")
		# print(f"(timestamps - timestamps[0]): {(timestamps - timestamps[0])}")
		# print(f"time_idx: {time_idx}")
		# Handle timestamps that are not floored and would exceed the allowed index after to-index conversion.
		time_idx[time_idx >= self.num_time_bins] = self.num_time_bins - 1

		spike_tensor[polarities.long(), locations[:, 1].long(), locations[:, 0].long(), time_idx.long()] = 1

		return spike_tensor
