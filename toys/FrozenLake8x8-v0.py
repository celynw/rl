#!/usr/bin/env python3
import colored_traceback.auto

import os
import base64
from pathlib import Path

from IPython import display as ipythondisplay
import gym
# from stable_baselines3 import PPO2
from stable_baselines3 import PPO
# from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from rich import print, inspect

# ==================================================================================================
def main():
	env = gym.make("CartPole-v1")
	# env = gym.make("FrozenLake8x8-v0")
	# model = PPO2("MlpPolicy", env, verbose=1)
	model = PPO(MlpPolicy, env, verbose=1)
	# model = A2C("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=10000)
	model.save(Path("checkpoints") / "ppo_cartpole.ckpt")
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
	print(f"mean_reward: {mean_reward:.2f} ±{std_reward:.2f}")
	record_video("CartPole-v1", model, video_length=500, prefix="ppo-cartpole")
	show_videos(Path("videos"), prefix="ppo")

	# obs = env.reset()
	# for i in range(1000):
	# 	action, _state = model.predict(obs, deterministic=True)
	# 	obs, reward, done, info = env.step(action)
	# 	env.render()
	# 	if done:
	# 		sobs = env.reset()


# ==================================================================================================
def record_video(env_id, model, video_length=500, prefix="", video_folder: Path = Path("videos")):
	"""
	:param env_id: (str)
	:param model: (RL model)
	:param video_length: (int)
	:param prefix: (str)
	:param video_folder: (str)
	"""
	eval_env = DummyVecEnv([lambda: gym.make(env_id)])
	# Start the video at step=0 and record 500 steps
	eval_env = VecVideoRecorder(eval_env, video_folder=str(video_folder),
								record_video_trigger=lambda step: step == 0, video_length=video_length,
								name_prefix=prefix)

	obs = eval_env.reset()
	for _ in range(video_length):
		action, _ = model.predict(obs)
		obs, _, _, _ = eval_env.step(action)

	# Close the video recorder
	eval_env.close()


# ==================================================================================================
def show_videos(video_path: Path = Path(""), prefix=""):
	"""
	Taken from https://github.com/eleurent/highway-env

	:param video_path: (str) Path to the folder containing videos
	:param prefix: (str) Filter the video, showing only the only starting with this prefix
	"""
	html = []
	for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
		video_b64 = base64.b64encode(mp4.read_bytes())
		html.append(
		"""<video alt="{}" autoplay loop controls style="height: 400px;">
			<source src="data:video/mp4;base64,{}" type="video/mp4" />
		</video>""".format(mp4, video_b64.decode("ascii")))
	ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


# ==================================================================================================
if __name__ == "__main__":
	# os.system("Xvfb :1 -screen 0 1024x768x24 &")
	# os.environ["DISPLAY"] = ":1"
	main()
