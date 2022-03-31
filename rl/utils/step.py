from enum import Enum

# ==================================================================================================
class Step(Enum):
	"""Prevents typos when using train, val or test anywhere."""
	TRAIN = "train"
	VAL = "val"
	TEST = "test"
	# ----------------------------------------------------------------------------------------------
	def __str__(self):
		return self.value
