from abc import abstractmethod
from pathlib import Path
import time
import argparse
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from rich import print, inspect

from esimcpp import SimulatorBridge
from rl.environments.utils import SpikeRepresentationGenerator

# ==================================================================================================
class EventEnv(gym.Env):
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

		self.bridge = SimulatorBridge(
			0.5, # Contrast threshold (positive): double contrast_threshold_pos
			0.5, # Contrast threshold (negative): double contrast_threshold_neg
			0.0, # Standard deviation of contrast threshold (positive): double contrast_threshold_sigma_pos = 0.021
			0.0, # Standard deviation of contrast threshold (negative): double contrast_threshold_sigma_neg = 0.021
			0, # Refractory period (time during which a pixel cannot fire events just after it fired one), in nanoseconds: int64_t refractory_period_ns
			True, # Whether to convert images to log images in the preprocessing step: const bool use_log_image
			0.001, # Epsilon value used to convert images to log: L = log(eps + I / 255.0): const double log_eps
			False, # Whether to simulate color events or not (default: false): const bool simulate_color_events
			# const double exposure_time_ms = 10.0, # Exposure time in milliseconds, used to simulate motion blur
			# const bool anonymous = false, # Whether to set a random number after the /ros_publisher node name (default: false)
			# const int32_t random_seed = 0 # Random seed used to generate the trajectories. If set to 0 the current time(0) is taken as seed.
		)

		self.generator = SpikeRepresentationGenerator(height, width, args.tsamples)
		self.fps = self.metadata["render_fps"]
		self.event_image = event_image
		self.time = 0.0
		self.events = None # Placeholder until messages arrive
		self.updatedPolicy = False # Used for logging whenever the policy is updated

		# FIX: I should normalise my observation space (well, both), but not sure how to for event tensor
		if self.event_image:
			self.shape = [2, height, width]
		else:
			self.shape = [2, args.tsamples, height, width]
		self.observation_space = spaces.Box(low=0, high=1, shape=self.shape, dtype=np.uint8)

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
		Resets the environment.

		Args:
			seed (int, optional): The seed that is used to initialize the environment's PRNG. Defaults to None.
			options (dict, optional): Additional information to specify how the environment is reset. Defaults to None.

		Returns:
			tuple[np.ndarray, Optional[dict]]: First observation and optionally info about the step.
		"""
		super().reset(seed=seed, options=options) # NOTE: Not using the output

		# Initialise ESIM; Need two frames to get a difference to generate events. The first should be all zero
		self.observe()
		self.observe()

		self.iter = 0

		return torch.zeros(*self.shape, dtype=torch.uint8).numpy(), self.get_info()

	# ----------------------------------------------------------------------------------------------
	def observe(self, rgb: Optional[np.ndarray] = None) -> Optional[torch.Tensor]:
		"""
		TODO Renders the AI gym environment, pushes it through ROS and presents the event observations.

		Args:
			rgb (np.ndarray, optional): RGB prerendered frame. Defaults to None.

		Returns:
			torch.Tensor: Observation as an event tensor.
		"""
		if rgb is None:
			rgb = self.render() # type: ignore
		rgb = self.resize(rgb) # type: ignore

		observation = self.events_to_tensor(self.bridge.img2events(rgb, int(self.time * 1e9)))
		self.time += (1 / self.fps)
		if self.event_image:
			observation = observation.sum(1) # Just sum the events for each pixel

		return observation

	# ----------------------------------------------------------------------------------------------
	def events_to_tensor(self, events: list) -> torch.Tensor:
		"""
		Processes the raw events into an event tensor.

		Args:
			events (list[Event]): List of raw event messages.

		Returns:
			torch.Tensor: Event tensor.
		"""
		polarities = torch.tensor([e.pol for e in events])
		coords = torch.tensor([[e.x, e.y] for e in events])
		stamps = torch.tensor([e.t / 1e9 for e in events], dtype=torch.double)
		assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

		event_tensor = self.generator.getSlayerSpikeTensor(polarities, coords, stamps)
		event_tensor = event_tensor.permute(0, 3, 1, 2) # CHWT -> CDHW

		return event_tensor

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
