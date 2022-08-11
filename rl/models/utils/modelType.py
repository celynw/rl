from enum import Enum

# ==================================================================================================
class ModelType(Enum):
	"""Prevents typos when using max or min anywhere."""
	DISCRETE = "discrete"
	VISUAL = "visual"
	# ----------------------------------------------------------------------------------------------
	def __str__(self):
		return self.value
