#!/usr/bin/env python3
import argparse
from argparse import _StoreTrueAction
from gettext import gettext as _

# ==================================================================================================
class StoreSuppress(_StoreTrueAction):
	"""
	Custom action to suppress already defined positional arguments.
	CAVEAT: This argument must be AFTER the arguments to suppress on the command line!

	Specify an additional argument `suppress: list[str]` to `argparse.add_argument`.
	"""
	# ----------------------------------------------------------------------------------------------
	def __init__(self, option_strings, dest, default: bool = False, required: bool = False, suppress: list[str] = [], help=None):
		self.suppress = suppress
		super().__init__(option_strings=option_strings, dest=dest, default=default, required=required, help=help)

	# ----------------------------------------------------------------------------------------------
	def __call__(self, parser, namespace, values, option_string=None):
		super().__call__(parser=parser, namespace=namespace, values=values, option_string=option_string)
		dest_values = []
		delete_indices = []
		required_actions = []
		for i, action in enumerate(parser._actions):
			print(action)
			if action.dest in self.suppress:
				dest_value = getattr(namespace, action.dest)
				if dest_value is not None:
					print(f"Store {action.dest}")
					dest_values.append(getattr(namespace, action.dest))
					print(f" -> {dest_values}")
				delete_indices.append(i)
				# setattr(namespace, action.dest, None)
				delattr(namespace, action.dest)
			elif len(dest_values) > 0:
				if action.option_strings:
					continue
				print(f"Pop {dest_values[0]} into {action.dest}")
				dest_value = dest_values.pop(0)
				setattr(namespace, action.dest, dest_value)
				if dest_value is None:
					required_actions.append(argparse._get_action_name(action))
					parser.error(_('the following arguments are required: %s') %
							', '.join(required_actions))
				else:
					action.required = False # To circumvent

		# Remove indices from list
		indicesList = sorted(delete_indices, reverse=True)
		for indx in indicesList:
			if indx < len(parser._actions):
				# print(f"deleting {parser._actions[indx]}")
				parser._actions.pop(indx)
				# del parser._actions[indx]

		# Error if there were extra arguments given
		if dest_values:
			# NOTE: Possibly cannot see all the additional arguments given
			# NOTE: Cannot see original argparse errors which occur after this
			msg = _('unrecognized argument: %s')
			parser.error(msg % ' '.join(dest_values))


# ==================================================================================================
if __name__ == "__main__":
	from rich import print

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("project", type=str, help="Project")
	parser.add_argument("name", type=str, help="Name")
	parser.add_argument("temp", type=str, help="Temporary positional!")
	parser.add_argument("-d", "--dryrun", action=StoreSuppress, suppress=["project", "name"], help="Dry run, overrides requirement `project` and `name` args")
	parser.add_argument("-f", "--fail", type=int, help="Extra arg")

	print(parser.parse_args())
