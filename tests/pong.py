#!/usr/bin/env python3
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN

import rl.models.utils
import rl.models
from rl.environments.utils import SkipCutscenesPong

config = {
	"env_name": "PongNoFrameskip-v4",
	"num_envs": 8,
	"total_timesteps": int(10e6),
	"seed": 4089164106,
}

run = wandb.init(
	project="Pong-reproduce",
	config=config,
	sync_tensorboard=True, # Auto-upload tensorboard metrics to wandb
	# monitor_gym=True, # Auto-upload the videos of agents playing the game
	save_code=True, # Save the code to W&B
)

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=8 => 8 environments)
env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"])

print("ENV ACTION SPACE: ", env.action_space.n)

env = SkipCutscenesPong(env)
env = VecFrameStack(env, n_stack=4)
# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)

# env.state_space # rl compatibility

# https://github.com/DLR-RM/rl-trained-agents/blob/10a9c31e806820d59b20d8b85ca67090338ea912/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4/config.yml
model = PPO(
# model = rl.models.utils.PPO(
	policy="CnnPolicy",
	# policy=rl.models.utils.ActorCriticPolicy,
	# policy_kwargs=dict(features_extractor_class=rl.models.NatureCNN),
	policy_kwargs=dict(features_extractor_class=NatureCNN),
	env=env,
	batch_size=256,
	clip_range=0.1,
	ent_coef=0.01,
	gae_lambda=0.9,
	gamma=0.99,
	learning_rate=2.5e-4,
	max_grad_norm=0.5,
	n_epochs=4,
	n_steps=128,
	vf_coef=0.5,
	tensorboard_log=f"runs",
	verbose=1,
)

model.learn(
	total_timesteps = config["total_timesteps"],
	callback = [
		WandbCallback(
			gradient_save_freq=1000,
			model_save_path=f"models/{run.id}",
		),
		CheckpointCallback(
			save_freq=10000,
			save_path='./pong',
			name_prefix=config["env_name"]
		),
	]
)

model.save("ppo-PongNoFrameskip-v4.zip")
