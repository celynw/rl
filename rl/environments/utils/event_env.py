from abc import abstractmethod
from pathlib import Path
import time
import argparse
from typing import Union, Optional

import gym
from gym import spaces
import numpy as np
import torch
from rich import print, inspect

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray, Event # Requries ROS to be sourced

from rl.environments.utils import SpikeRepresentationGenerator

# ==================================================================================================
class EventEnv(gym.Env):
	debug = False # For ROS logging
	# ----------------------------------------------------------------------------------------------
	def __init__(self, width: int, height: int, args: argparse.Namespace, event_image: bool = False):
		"""
		Base class which handles the ROS side for event versions of AI gym environments.

		Args:
			width (int): Width of the event frame in pixels.
			height (int): Height of the event frame in pixels.
			args (argparse.Namespace): Parsed arguments, depends on which specific env we're using.
			event_image (bool, optional): Accuumlates events into an event image. Defaults to False.
		"""
		self.state_space = self.observation_space # For access later
		if args.fps is not None:
			self.metadata["render_fps"] = args.fps
		else:
			fps = self.metadata["render_fps"] # type: ignore

		rospy.init_node(Path(__file__).stem, anonymous=False, log_level=rospy.DEBUG if self.debug else rospy.INFO)
		self.bridge = CvBridge()
		self.pub_image = rospy.Publisher("image", Image, queue_size=10)
		self.sub_events = rospy.Subscriber("/cam0/events", EventArray, self.callback)
		self.generator = SpikeRepresentationGenerator(height, width, args.tsamples)
		self.fps = self.metadata["render_fps"]
		self.event_image = event_image
		self.time = rospy.Time.now() # Initial timestamp for events
		self.events = None # Placeholder until messages arrive
		self.connected = False # Waits for subscriber to listen to our published topic
		self.updatedPolicy = False # Used for logging whenever the policy is updated

		# FIX: I should normalise my observation space (well, both), but not sure how to for event tensor
		if self.event_image:
			self.shape = [2, height, width]
		else:
			self.shape = [2, args.tsamples, height, width]
		self.observation_space = spaces.Box(low=0, high=1, shape=self.shape, dtype=np.double)

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		group = parser.add_argument_group("Environment")
		group.add_argument("--fps", type=float, help="Frames per second of environment. Default is environment default", default=None)
		group.add_argument("--tsamples", type=int, default=6, help="How many time samples to use in event environments")

		return parser

	# ----------------------------------------------------------------------------------------------
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
		"""
		Resets the environment, and also the model (if defined).

		Args:
			seed (int, optional): The seed that is used to initialize the environment's PRNG. Defaults to None.
			options (dict, optional): Additional information to specify how the environment is reset. Defaults to None.

		Returns:
			tuple[np.ndarray, Optional[dict]]: First observation and optionally info about the step.
		"""
		super().reset(seed=seed, options=options) # NOTE: Not using the output
		info = self.get_info()

		self.observe(wait=False) # Initialise ESIM; Need two frames to get a difference to generate events
		self.observe()
		if self.model is not None and hasattr(self.model, "reset_env"):
			self.model.reset_env()

		return torch.zeros(*self.shape, dtype=torch.double).numpy(), info

	# ----------------------------------------------------------------------------------------------
	def observe(self, rgb: Optional[np.ndarray] = None, wait: bool = True) -> Optional[torch.Tensor]:
		"""
		Renders the AI gym environment, pushes it through ROS and presents the event observations.

		Args:
			rgb (np.ndarray, optional): RGB prerendered frame. Defaults to None.

		Returns:
			torch.Tensor: Observation as an event tensor.
		"""
		if rgb is None:
			rgb = self.render() # type: ignore
		rgb = self.resize(rgb) # type: ignore
		self.publish(rgb) # type: ignore

		if not wait:
			return

		observation = self.events_to_tensor(self.wait_for_events())
		if self.event_image:
			observation = observation.sum(1) # Just sum the events for each pixel

		return observation

	# ----------------------------------------------------------------------------------------------
	def wait_for_events(self) -> list[Event]:
		"""
		Waits until `self.events` is populated.

		Returns:
			list[Event]: The value of self.events: the events from the EventArray ROS message.
		"""
		while self.events is None:
			# pass
			if rospy.is_shutdown():
				print("Node is shutting down")
				break
				# quit(0)

		return self.events

	# ----------------------------------------------------------------------------------------------
	def publish(self, rgb: np.ndarray):
		"""
		Publishes the RGB image so that the events can be generated using the simulator.

		Args:
			rgb (np.ndarray): Original RGB observation from AI gym.
		"""
		msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
		# msg.header.frame_id = self.params.cam_frame
		msg.header.frame_id = "cam"
		msg.header.stamp = self.time
		self.events = None

		# Wait for subscriber. Can be used to reveal a problem with nodes or publishing/subscribing
		i = 0
		while not self.connected:
			print(f"Waiting for subscriber on '{self.pub_image.name}' topic ({i})")
			time.sleep(1)
			# Check that ESIM is active
			# FIX this doesn't account for _what_ is listening, it could be just a visualiser or rosbag!
			if self.pub_image.get_num_connections() > 0:
				self.connected = True
				print("Connected")
			if rospy.is_shutdown():
				print("Node is shutting down")
				break
			i += 1

		self.pub_image.publish(msg)
		self.time += rospy.Duration(nsecs=int((1 / self.fps) * 1e9))

	# ----------------------------------------------------------------------------------------------
	def events_to_tensor(self, events: list[Event]) -> torch.Tensor:
		"""
		Processes the raw events into an event tensor.

		Args:
			events (list[Event]): List of raw event messages.

		Returns:
			torch.Tensor: Event tensor.
		"""
		polarities = torch.tensor([e.polarity for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.ts.to_sec() for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		event_tensor = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		event_tensor = event_tensor.permute(0, 3, 1, 2) # CHWT -> CDHW

		return event_tensor

	# ----------------------------------------------------------------------------------------------
	def callback(self, msg: EventArray):
		"""
		ROS callback on receiving an EventArray message.
		Sets the class `self.events` to that data.

		Args:
			msg (EventArray): Raw ROS message from event simulator.
		"""
		self.events = msg.events

	# ----------------------------------------------------------------------------------------------
	def set_state(self, state: tuple):
		"""
		Manually set the state of the environment.
		Not to be used under normal training circumstances.

		Args:
			state (tuple): State of the particular environment.
		"""
		self.state = state

	# ----------------------------------------------------------------------------------------------
	def set_updatedPolicy(self):
		"""
		Set a boolean flag. Used for logging whenever the policy is updated.
		TODO move to model. It's only in the env for program accessibility...
		"""
		self.updatedPolicy = True

	# ----------------------------------------------------------------------------------------------
	def set_model(self, model):
		"""
		Link the model and the env, so we can call a model function when the env calls `reset()`.

		Args:
			model: Model (feature extractor) object.
		"""
		self.model = model

	# ----------------------------------------------------------------------------------------------
	def get_info(self) -> dict:
		"""
		Return a created dictionary for the step info.

		Returns:
			dict: Key-value pairs for the step info.
		"""
		return {
			"state": self.state, # Used later for bootstrap loss
			"updatedPolicy": int(self.updatedPolicy),
		}

	# ----------------------------------------------------------------------------------------------
	@abstractmethod
	def resize(self, rgb: np.ndarray):
		"""
		Base method to downsize the original RGB observations for more bitesize event tensors.

		Args:
			rgb (np.ndarray): Original RGB render of the observation, probably as OpenCV image.
		"""
		raise NotImplementedError
