from enum import Enum

# ==================================================================================================
class Dir(Enum):
	"""Prevents typos when using max or min anywhere."""
	MIN = "min"
	MAX = "max"
	# ----------------------------------------------------------------------------------------------
	def __str__(self):
		return self.value
