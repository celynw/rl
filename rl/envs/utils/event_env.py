import time
from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional

import gym
import numpy as np
import torch

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray, Event

from rl.envs.utils import SpikeRepresentationGenerator

# ==================================================================================================
class EventEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, screen_width: int, screen_height: int, init_ros: bool = True, tsamples: int = 10, event_image: bool = False):
		self.init_ros = init_ros
		self.tsamples = tsamples
		self.event_image = event_image
		self.connected = False
		self.updatedPolicy = False
		self.model = None

		name = Path(__file__).stem# if name is None else None
		debug = False
		if self.init_ros:
			rospy.init_node(name, anonymous=False, log_level=rospy.DEBUG if debug else rospy.INFO)
		self.bridge = CvBridge()
		self.pub_image = rospy.Publisher("image", Image, queue_size=10)
		self.sub_events = rospy.Subscriber("/cam0/events", EventArray, self.callback)
		self.events = None
		self.generator = SpikeRepresentationGenerator(screen_height, screen_width, self.tsamples)
		if self.init_ros:
			self.time = rospy.Time.now()

	# ----------------------------------------------------------------------------------------------
	def set_model(self, model):
		self.model = model

	# ----------------------------------------------------------------------------------------------
	def set_state(self, state):
		self.state = state

	# ----------------------------------------------------------------------------------------------
	def set_updatedPolicy(self):
		self.updatedPolicy = True

	# ----------------------------------------------------------------------------------------------
	@abstractmethod
	def resize(self, rgb):
		return

	# ----------------------------------------------------------------------------------------------
	def get_events(self, wait: bool = True) -> Optional[torch.Tensor]:
		rgb = self.render("rgb_array")
		rgb = self.resize(rgb)

		if self.init_ros:
			self.publish(rgb)
		if wait:
			return self.tensorify_edenn(self.wait_for_events())
		else:
			return

	# ----------------------------------------------------------------------------------------------
	def publish(self, obs):
		try:
			msg = self.bridge.cv2_to_imgmsg(obs, encoding="rgb8")
		except CvBridgeError as e:
			print(e)
		# msg.header.frame_id = self.params.cam_frame
		msg.header.frame_id = "cam"
		msg.header.stamp = self.time
		self.events = None

		i = 0
		while not self.connected:
			print(f"Waiting for subscriber on '{self.pub_image.name}' topic ({i})")
			# Check that ESIM is active
			# FIX this doesn't account for _what_ is listening, it could be just a visualiser or rosbag!
			if self.pub_image.get_num_connections() > 0:
				self.connected = True
				print("Connected")
			i += 1
			time.sleep(1)

		self.pub_image.publish(msg)
		self.time += rospy.Duration(nsecs=int((1 / 30) * 1e9))

	# ----------------------------------------------------------------------------------------------
	def wait_for_events(self):
		while self.events is None:
			pass

		if rospy.is_shutdown():
			print("Node is shutting down")
			quit(0)

		return self.events

	# ----------------------------------------------------------------------------------------------
	def tensorify_accumulate(self, events: list[Event]) -> torch.Tensor:
		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		event_tensor = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		event_tensor = event_tensor.permute(0, 3, 1, 2) # CHWT -> CDHW

		return event_tensor

	# ----------------------------------------------------------------------------------------------
	def tensorify_evflownet(self, events: list[Event]) -> torch.Tensor:
		# 4-channels * H * W:
		# - 0: number of positive
		# - 1: number of negative
		# - 2: most recent timestamp of positive
		# - 3: most recent timestamp of negative
		# I'm assuming the events list is sorted! By timestamp ascending

		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		spike_tensor = torch.zeros((4, self.screen_height_, self.screen_width_))
		if len(stamps) == 0:
			return spike_tensor # Empty tensor, don't raise an error
		# spike_tensor[polarities.long(), coords[:, 1].long(), coords[:, 0].long()] += 1
		# # But what about the timestamps?
		# FIX better to use start time which is the start time of this frame, not first event!
		first_timestamp = stamps[0]
		for event in events:
			# Increment number of events
			spike_tensor[int(event.polarity), event.y, event.x] += 1
			# Update timestamp
			ts = event.ts.to_sec() - first_timestamp
			if ts > spike_tensor[int(event.polarity) + 2, event.y, event.x]:
				spike_tensor[int(event.polarity) + 2, event.y, event.x] = ts

		return spike_tensor

	# ----------------------------------------------------------------------------------------------
	def tensorify_edenn(self, events: list[Event]) -> torch.Tensor:
		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		event_tensor = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		event_tensor = event_tensor.permute(0, 3, 1, 2) # CHWT -> CDHW

		return event_tensor

	# ----------------------------------------------------------------------------------------------
	def callback(self, msg):
		self.events = msg.events
