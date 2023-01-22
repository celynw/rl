from typing import Optional

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder as gym_VideoRecorder
from gymnasium import logger
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images
import numpy as np

from rl.environments.utils import add_event_image_channel

# ==================================================================================================
class VideoRecorder(gym_VideoRecorder):
	# ----------------------------------------------------------------------------------------------
	def __init__(self,
		env,
		path: Optional[str] = None,
		metadata: Optional[dict] = None,
		enabled: bool = True,
		base_path: Optional[str] = None,
		disable_logger: bool = False,
		render_events: bool = False,
		sum_events: bool = True,
	):
		super().__init__(env, path, metadata, enabled, base_path, disable_logger)
		self.render_events = render_events
		self.sum_events = sum_events
	# ----------------------------------------------------------------------------------------------
	def capture_frame(self):
		"""Render the given `env` and add the resulting frame to the video."""
		if self.render_events:
			if isinstance(self.env, DummyVecEnv):
				if self.sum_events:
					images = [np.transpose(add_event_image_channel(np.asarray(env.events.sum(1)) * 255), (1, 2, 0)) for env in self.env.envs]
				else:
					images = [np.transpose(add_event_image_channel(np.asarray(env.events) * 255), (1, 2, 0)) for env in self.env.envs]
				# Create a big image by tiling images from subprocesses
				frame = tile_images(images)
			else:
				frame = self.env.events.sum(1) # Hope this is being set in the last env.observe()
		else:
			frame = self.env.render()

		if isinstance(frame, list):
			self.render_history += frame
			frame = frame[-1]

		if not self.functional:
			return
		if self._closed:
			logger.warn(
				"The video recorder has been closed and no frames will be captured anymore."
			)
			return
		logger.debug("Capturing video frame: path=%s", self.path)

		if frame is None:
			if self._async:
				return
			else:
				# Indicates a bug in the environment: don't want to raise
				# an error here.
				logger.warn(
					"Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
					f"disabled: path={self.path} metadata_path={self.metadata_path}"
				)
				self.broken = True
		else:
			self.recorded_frames.append(frame)
