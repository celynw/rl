#!/usr/bin/env python3
import argparse
from argparse import _StoreTrueAction

# ==================================================================================================
class StoreSuppress(_StoreTrueAction):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, option_strings, dest, default: bool = False, required: bool = False, suppress: list[str] = [], help=None):
		self.suppress = suppress
		super().__init__(option_strings=option_strings, dest=dest, default=default, required=required, help=help)

	# ----------------------------------------------------------------------------------------------
	def __call__(self, parser, namespace, values, option_string=None):
		super().__call__(parser=parser, namespace=namespace, values=values, option_string=option_string)
		for suppress in self.suppress:
			for i, action in enumerate(parser._actions):
				if action.dest == suppress:
					del parser._actions[i]


# ==================================================================================================
if __name__ == "__main__":
	from rich import print

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
	parser.add_argument("project", type=str, help="Project")
	parser.add_argument("name", type=str, help="Name")
	parser.add_argument("-d", "--dryrun", action=StoreSuppress, suppress=["project", "name"], help="Dry run, overrides requirement `project` and `name` args")

	print(parser.parse_args())
