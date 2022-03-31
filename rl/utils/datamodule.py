from pytorch_lightning import LightningDataModule

# ==================================================================================================
class DataModule(LightningDataModule):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, cpu_only: list[str] = []):
		"""LightningDataModule, adding the option to not put certain batch members on the device."""
		super().__init__()
		self.cpu_only = cpu_only

	# ----------------------------------------------------------------------------------------------
	def transfer_batch_to_device(self, batch, device, dataloader_idx):
		if isinstance(batch, dict):
			for key in batch:
				if key not in self.cpu_only:
					batch[key] = super().transfer_batch_to_device(batch[key], device, dataloader_idx)

			return batch
		else:
			return super().transfer_batch_to_device(batch, device, dataloader_idx)
