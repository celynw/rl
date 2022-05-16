#!/usr/bin/env python
"""Template rospy node."""
import colored_traceback.auto
import rospy
import argparse
from pathlib import Path
from munch import Munch
from rich import print, inspect

from dvs_msgs.msg import EventArray

# ==================================================================================================
class Node():
	# ----------------------------------------------------------------------------------------------
	def __init__(self, name=None, anonymous=True, debug=False, init=True):
		if init:
			name = Path(__file__).stem if name is None else None
			rospy.init_node(name, anonymous=anonymous, log_level=rospy.DEBUG if debug else rospy.INFO)
		self.parse_params()
		if init:
			self.setup()
		self.register()
		if init:
			self.start()

	# ----------------------------------------------------------------------------------------------
	def parse_params(self):
		self.params = Munch()

	# ----------------------------------------------------------------------------------------------
	def register(self):
		self.sub = Munch()
		self.sub.image = rospy.Subscriber("events", EventArray, self.events_cb, queue_size=1)

		self.pub = Munch()

	# ----------------------------------------------------------------------------------------------
	def setup(self):
		self.start_time = None

	# ----------------------------------------------------------------------------------------------
	def start(self):
		while not rospy.is_shutdown():
			pass

	# ----------------------------------------------------------------------------------------------
	def events_cb(self, msg):
		if self.start_time is None:
			self.start_time = msg.header.stamp.to_sec()
		times = set()
		for event in msg.events:
			times.add(event.ts.to_sec())
		print(f"{msg.header.stamp.to_sec() - self.start_time:.3f} [{len(times)} unique / {len(msg.events)} total]")

	# ----------------------------------------------------------------------------------------------
	@classmethod
	def get_epilog(cls):
		node = cls(init=False)
		subList = [f"    * {sub.name} [{sub.data_class._type}]" for sub in node.sub.values()]
		pubList = [f"    * {pub.name} [{pub.data_class._type}]" for pub in node.pub.values()]
		paramList = [f"    * {k}: {v}" for k, v in node.params.items()]
		for sub in node.sub.values():
			sub.unregister()
		for pub in node.sub.values():
			pub.unregister()
		del(node)

		epilog = ""
		if subList:
			epilog += "Subscribed topics:\n"
			epilog += "\n".join(subList) + "\n"
		if pubList:
			epilog += "Published topics:\n"
			epilog += "\n".join(pubList) + "\n"
		if paramList:
			epilog += "Params:\n"
			epilog += "\n".join(paramList) + "\n"

		return epilog


# ==================================================================================================
if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=Node.get_epilog())
	parser.add_argument("-a", "--anonymous", action="store_true", help="Run anonymously, i.e. set a random name")
	parser.add_argument("-d", "--debug", action="store_true", help="Set log level to DEBUG")
	args = parser.parse_args(rospy.myargv()[1:])
	try:
		Node(anonymous=args.anonymous, debug=args.debug)
	except rospy.ROSInterruptException:
		pass
