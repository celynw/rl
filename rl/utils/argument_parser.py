import argparse
import textwrap

# ==================================================================================================
class ArgumentParser(argparse.ArgumentParser):
	"""Improves help printing with partially assigned arguments."""
	# ----------------------------------------------------------------------------------------------
	def error(self, message: str):
		"""
		Print the problems with the arguments.
		Also prints a helpful message about model types.

		Args:
			message (str): Message to print
		"""
		# self.print_usage(sys.stderr)
		self.print_help() # Also prints usage
		args = {"prog": self.prog, "message": message}
		self.exit(2, textwrap.dedent((f"""\
			{args['prog']}: error: {args['message']}
			More help options will be provided when specifying the model and dataset types.
		""")))
