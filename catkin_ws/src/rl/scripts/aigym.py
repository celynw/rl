#!/usr/bin/env python
"""Runs an AI gym environment and publishes the screen to an image topic."""
import colored_traceback.auto
import rospy
import argparse
from pathlib import Path
from munch import Munch

import tf2_ros
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import gymnasium as gym
from rich import print, inspect

import rl # Registers custom environments

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
		self.params.cam_frame = rospy.get_param("~cam_frame", "cam")
		self.params.rate = rospy.get_param("~rate", 10.0)
		self.params.iters = rospy.get_param("~iters", 0)
		self.params.render = rospy.get_param("~render", False)

	# ----------------------------------------------------------------------------------------------
	def register(self):
		self.sub = Munch()

		self.pub = Munch()
		self.pub.image = rospy.Publisher("image", Image, queue_size=10)

	# ----------------------------------------------------------------------------------------------
	def setup(self):
		self.bridge = CvBridge()

		self.rate = rospy.Rate(self.params.rate)

		self.env = gym.make("CartPole-contrast-v1")
		self.env.reset()

	# ----------------------------------------------------------------------------------------------
	def start(self):
		done = False
		iters = 0
		while not rospy.is_shutdown():
			if self.params.iters and iters >= self.params.iters:
				break
			action = 1
			if done:
				obs = self.env.reset()
				done = False
			else:
				obs, reward, done, info = self.env.step(action)

			if self.params.render:
				self.env.render()
			out = self.env.render(mode="rgb_array")
			try:
				msg = self.bridge.cv2_to_imgmsg(out, encoding="rgb8")
			except CvBridgeError as e:
				print(e)
			msg.header.frame_id = self.params.cam_frame
			msg.header.stamp = rospy.Time.now()
			self.pub.image.publish(msg)
			self.rate.sleep()

			iters += 1

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
