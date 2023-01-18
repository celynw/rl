import os
from typing import Callable

from stable_baselines3.common.vec_env import VecVideoRecorder as SB3_VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from rl.environments.utils import VideoRecorder, get_base_envs

# ==================================================================================================
# Adding another env.render_mode is too deeply ingrained into SB3 and gym, ffs...
class VecVideoRecorder(SB3_VecVideoRecorder):
	"""
	Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
	It requires ffmpeg or avconv to be installed on the machine.

	:param venv:
	:param video_folder: Where to save videos
	:param record_video_trigger: Function that defines when to start recording.
		The function takes the current number of step,
		and returns whether we should start recording or not.
	:param video_length: Length of recorded videos
	:param name_prefix: Prefix to the video name
	:param events: Render event stream as image
	"""
	# ----------------------------------------------------------------------------------------------
	# def __init__(self, *args, render_events: bool = False, **kwargs):
	def __init__(
		self,
		venv: VecEnv,
		video_folder: str,
		record_video_trigger: Callable[[int], bool],
		video_length: int = 200,
		name_prefix: str = "rl-video",
		render_events: bool = False,
	):
		# super().__init__(*args, **kwargs)
		VecEnvWrapper.__init__(self, venv)

		self.env = venv
		# Temp variable to retrieve metadata
		temp_env = venv

		# Unwrap to retrieve metadata dict
		# that will be used by gym recorder
		while isinstance(temp_env, VecEnvWrapper):
			temp_env = temp_env.venv

		if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
			metadata = temp_env.get_attr("metadata")[0]
		else:
			metadata = temp_env.metadata

		self.env.metadata = metadata
		self.env.render_mode = get_base_envs(self.env)[0].render_mode
		assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

		self.record_video_trigger = record_video_trigger
		self.video_recorder = None

		self.video_folder = os.path.abspath(video_folder)
		# Create output folder if needed
		os.makedirs(self.video_folder, exist_ok=True)

		self.name_prefix = name_prefix
		self.step_id = 0
		self.video_length = video_length

		self.recording = False
		self.recorded_frames = 0

		# self.env.render_mode = get_base_envs(self.env)[0].render_mode

		self.render_events = render_events
		if render_events:
			self.name_prefix += "-events"

	# ----------------------------------------------------------------------------------------------
	def start_video_recorder(self) -> None:
		self.close_video_recorder()

		video_name = f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
		base_path = os.path.join(self.video_folder, video_name)
		self.video_recorder = VideoRecorder(
			env=self.env, base_path=base_path, metadata={"step_id": self.step_id}, render_events=self.render_events
		)

		self.video_recorder.capture_frame()
		self.recorded_frames = 1
		self.recording = True
